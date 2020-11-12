import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Any, Union, Type, Tuple, Callable, Optional
from unittest import mock

import torch
from torch.serialization import default_restore_location
import fairseq
from fairseq.models.transformer import TransformerModel

PATH_MODEL_REPO = Path('...')


# Mock missing fairseq's logging
class AverageMeter:
    def __init__(self):
        pass


class TimeMeter:
    def __init__(self):
        pass


class StopwatchMeter:
    def __init__(self):
        pass


class Meters:
    def __init__(self):
        self.AverageMeter = AverageMeter
        self.TimeMeter = TimeMeter
        self.StopwatchMeter = StopwatchMeter


sys.modules['fairseq.logging'] = mock.Mock()
sys.modules['fairseq.logging.meters'] = Meters()


class PathManager:
    """
    Wrapper for insulating OSS I/O (using Python builtin operations) from
    fvcore's PathManager abstraction (for transparently handling various
    internal backends).
    """
    @staticmethod
    def open(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        return open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""
    from fairseq import models, registry, tasks

    # add optimizer_history
    if "optimizer_history" not in state:
        state["optimizer_history"] = [{
            "criterion_name": "CrossEntropyCriterion",
            "best_loss": state["best_loss"]
        }]
        state["last_optimizer_state"] = state["optimizer"]
        del state["optimizer"]
        del state["best_loss"]
    # move extra_state into sub-dictionary
    if "epoch" in state and "extra_state" not in state:
        state["extra_state"] = {
            "epoch": state["epoch"],
            "batch_offset": state["batch_offset"],
            "val_loss": state["val_loss"],
        }
        del state["epoch"]
        del state["batch_offset"]
        del state["val_loss"]
    # reduce optimizer history's memory usage (only keep the last state)
    if "optimizer" in state["optimizer_history"][-1]:
        state["last_optimizer_state"] = state["optimizer_history"][-1][
            "optimizer"]
        for optim_hist in state["optimizer_history"]:
            del optim_hist["optimizer"]
    # record the optimizer class name
    if "optimizer_name" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["optimizer_name"] = "FairseqNAG"
    # move best_loss into lr_scheduler_state
    if "lr_scheduler_state" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["lr_scheduler_state"] = {
            "best": state["optimizer_history"][-1]["best_loss"]
        }
        del state["optimizer_history"][-1]["best_loss"]
    # keep track of number of updates
    if "num_updates" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["num_updates"] = 0
    # old model checkpoints may not have separate source/target positions
    if hasattr(state["args"], "max_positions") and not hasattr(
            state["args"], "max_source_positions"):
        state["args"].max_source_positions = state["args"].max_positions
        state["args"].max_target_positions = state["args"].max_positions
    # use stateful training data iterator
    if "train_iterator" not in state["extra_state"]:
        state["extra_state"]["train_iterator"] = {
            "epoch": state["extra_state"]["epoch"],
            "iterations_in_epoch": state["extra_state"].get("batch_offset", 0),
        }
    # default to translation task
    if not hasattr(state["args"], "task"):
        state["args"].task = "translation"
    # --raw-text and --lazy-load are deprecated
    if getattr(state["args"], "raw_text", False):
        state["args"].dataset_impl = "raw"
    elif getattr(state["args"], "lazy_load", False):
        state["args"].dataset_impl = "lazy"
    # epochs start at 1
    if state["extra_state"]["train_iterator"] is not None:
        state["extra_state"]["train_iterator"]["epoch"] = max(
            state["extra_state"]["train_iterator"].get("epoch", 1),
            1,
        )

    # set any missing default values in the task, model or other registries
    registry.set_defaults(state["args"],
                          tasks.TASK_REGISTRY[state["args"].task])
    registry.set_defaults(state["args"],
                          models.ARCH_MODEL_REGISTRY[state["args"].arch])
    for registry_name, REGISTRY in registry.REGISTRIES.items():
        choice = getattr(state["args"], registry_name, None)
        if choice is not None:
            cls = REGISTRY["registry"][choice]
            registry.set_defaults(state["args"], cls)

    return state


def load_checkpoint_to_cpu(path, arg_overrides=None):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility)."""
    with PathManager.open(path, "rb") as f:
        state = torch.load(
            f, map_location=lambda s, l: default_restore_location(s, "cpu"))

    args = state["args"]
    if arg_overrides is not None:
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)
    state = _upgrade_state_dict(state)
    return state


# pylint: disable=no-member
class Search(object):
    def __init__(self, tgt_dict):
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.scores_buf = None
        self.indices_buf = None
        self.beams_buf = None

    def _init_buffers(self, t):
        if self.scores_buf is None:
            self.scores_buf = t.new()
            self.indices_buf = torch.LongTensor().to(device=t.device)
            self.beams_buf = torch.LongTensor().to(device=t.device)

    def step(self, step, lprobs, scores):
        raise NotImplementedError

    def set_src_lengths(self, src_lengths):
        self.src_lengths = src_lengths


# pylint: disable=no-member
class BeamSearch(Search):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    def step(self, step, lprobs, scores):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))

        torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
            out=(self.scores_buf, self.indices_buf),
        )
        self.beams_buf = self.indices_buf // vocab_size
        self.indices_buf.fmod_(vocab_size)
        return self.scores_buf, self.indices_buf, self.beams_buf


# Monkeypatch broken fairseq functions/classes
fairseq.checkpoint_utils._upgrade_state_dict = _upgrade_state_dict
fairseq.checkpoint_utils.load_checkpoint_to_cpu = load_checkpoint_to_cpu
fairseq.search.BeamSearch = BeamSearch

class Translator:
    def __init__(self, source: str, target: str) -> None:
      pass
