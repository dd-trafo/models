from pathlib import Path
from typing import Dict, List, Any, Union, Type, Tuple, Callable

from fairseq.models.transformer import TransformerModel

from . import PATH_MODEL_REPO


class Translator:
    def __init__(self, source: str, target: str) -> None:

        path = PATH_MODEL_REPO / 'translation'
        if (source == 'EN') and (target == 'DE'):
            subfolder = 'en2de_v1'
        elif (source == 'DE') and (target == 'EN'):
            subfolder = 'de2en_v1'
        elif (source == 'EN') and (target == 'FR'):
            subfolder = 'en2fr_v2'
        elif (source == 'FR') and (target == 'EN'):
            subfolder = 'fr2en_v2'
        elif (source == 'EN') and (target == 'IT'):
            subfolder = 'en2it_v1'
        elif (source == 'IT') and (target == 'EN'):
            subfolder = 'it2en_v1'
        else:
            raise NotImplementedError(
                f'No model available for translation from "{source}" to "{target}".'
            )

        self.model = TransformerModel.from_pretrained(
            path / subfolder,
            checkpoint_file='checkpoint_average_last10.pt',
            bpe='sentencepiece',
            bpe_codes='sentencepiece.bpe.model',
        )

    def translate(
        self,
        txt: str,
        beam: int = 1,
        sample: bool = False,
    ) -> str:

        # Strip First, remove whitespace
        txt = txt.strip()

        # If nothing is left, stop here
        if len(txt) == 0:
            if beam == 1:
                return ''
            else:
                return ['']

        # If `txt` does not have a sentence boundary, add it
        if txt[-1] in ['.', '?', '!']:
            has_sentence_boundary = True
        else:
            has_sentence_boundary = False
            txt = txt + '.'

        if beam == 1:
            translations = [self.model.translate(txt)]
        else:
            # Preprocess `txt`, i.e. tokenization
            txt_tokens = self.model.tokenize(txt)
            txt_bpe = self.model.apply_bpe(txt_tokens)
            txt_bin = self.model.binarize(txt_bpe)

            # Sampling affects results from beam search
            if sample:
                translations_bin = self.model.generate(
                    txt_bin,
                    beam=beam,
                    sampling=sample,
                    sampling_topk=3,
                )
            else:
                translations_bin = self.model.generate(
                    txt_bin,
                    beam=beam,
                )

            # Postprocess, i.e. detokenization
            translations = list()
            for translation_bin in translations_bin:
                translation_bpe = self.model.string(translation_bin['tokens'])
                translation_tokens = self.model.remove_bpe(translation_bpe)
                translations.append(self.model.detokenize(translation_tokens))

            # Get unique translations
            translations = list(set(translations))

        for i in range(len(translations)):
            translations[i] = translations[i].strip()
            if not has_sentence_boundary:
                if translations[i][-1] == '.':
                    translations[i] = translations[i][:-1]

        if beam == 1:
            return translations[0]
        else:
            return translations
