import re
from typing import Dict, List, Union

import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer

from .translator import Translator


class Paraphraser:
    def __init__(
        self,
        type: str,
        source: Union[None, str] = None,
        proxy: Union[None, str] = None,
        path: Union[None, str] = None,
    ):
        """
        Paraphraser produces similar text using either roundtrip translation
        (fairseq BERT) or text generation of mask tokens (huggingface BART).
        """

        if type == 'translation':
            self.type = type
            self.model_forward = Translator(source=source, target=proxy)
            self.model_backward = Translator(source=proxy, target=source)
        elif type == 'generation':
            self.type = type
            if path is None:
                path = '.../bart-base'
            self.tokenizer = BartTokenizer.from_pretrained(path)
            self.model = BartForConditionalGeneration.from_pretrained(path)
            self.mask_token = self.tokenizer.mask_token
        else:
            raise NotImplementedError(
                '`type` must be `translation` or `generation`.')

    def paraphrase(
        self,
        txt: str,
        beam: int = 1,
    ):
        """
        Use roundtrip translation to produce similar sentences.
        """

        assert self.type == 'translation', 'Requires `type` == `translation`.'

        txt_proxy = self.model_forward.translate(txt, beam=1)
        return self.model_backward.translate(txt_proxy, beam=beam)

    def fill(
        self,
        txt,
        **kwargs,
    ):
        """
        Fill mask token with generated text.
        """

        assert self.type == 'generation', 'Requires `type` == `generation`.'

        inputs = self.tokenizer.encode(txt, return_tensors='pt')

        if 'max_length' not in kwargs:
            kwargs['max_length'] = len(inputs) + 50

        outputs = self.model.generate(inputs, **kwargs)

        txt = list()
        for output in outputs:
            txt.append(self.tokenizer.decode(output, skip_special_tokens=True))

        if len(txt) == 1:
            return txt[0]
        else:
            return list(set(txt))

    def _tokens_to_words(
        self,
        explanation: Dict,
    ):
        """
        Convert an explanation from token- to word-level, using minimum weight.
        """

        explanation_words = list()
        tokens = list()
        min_weight = np.inf

        # Iterate over tokens in explanation
        for item in explanation['weights']:

            token = item['token']

            if token.startswith('##'):
                # Token is word continuation
                token = token[2:]
                tokens.append(token)
                min_weight = min(min_weight, item['weight'])
            else:
                # Token starts a new word, thus write out previous word
                if len(tokens) > 0:
                    explanation_words.append({
                        'word': ''.join(tokens),
                        'weight': min_weight
                    })

                tokens = [token]
                min_weight = item['weight']

        # If exists, add remaining `tokens`
        if len(tokens) > 0:
            explanation_words.append({
                'word': ''.join(tokens),
                'weight': min_weight
            })

        return explanation_words

    def adaptive_paraphrase(
        self,
        explanation: Dict,
        weight_threshold: float = 0.0,
        do_mask_negative_weight: bool = True,
        **kwargs,
    ):
        """
        Generate text for masked tokens based on explanation.
        Tokens below `weight_threshold` are masked.
        """

        assert self.type == 'generation', 'Requires `type` == `generation`.'

        explanation_words = self._tokens_to_words(explanation)

        words = list()
        sign = 1.0 if do_mask_negative_weight else -1.0
        for item in explanation_words:
            if (sign * item['weight']) >= weight_threshold:
                words.append(item['word'])
            else:
                words.append(self.mask_token)

        txt = ' '.join(words)
        txt = re.sub(' ([.?!,;:])', r'\1', txt)
        txt = re.sub(f'({self.mask_token} )+', f'{self.mask_token} ', txt)

        if self.mask_token in txt:
            return self.fill(txt, **kwargs)
        else:
            if 'num_return_sequences' in kwargs:
                if kwargs['num_return_sequences'] > 1:
                    return [txt]
                else:
                    return txt
            else:
                return txt
