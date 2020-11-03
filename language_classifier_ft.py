from pathlib import Path

import fasttext

from . import PATH_MODEL_REPO


class LanguageClassifier:
    def __init__(
        self,
        languages=None,
    ) -> None:

        if languages is None:
            self.languages = languages
        else:
            self.languages = [language.upper() for language in languages]

        # Silence deprecation warning
        fasttext.FastText.eprint = lambda x: None

        # Download:
        # https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
        # - or -
        # https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
        self.model = fasttext.load_model(
            str(PATH_MODEL_REPO / 'fasttext' / 'lid.176.bin'))

    def classify(self, txt: str) -> str:
        try:
            result = self.model.predict(txt, k=1)[0][0][-2:].upper()
        except Exception as e:
            print(f'Error: {e}')
            result = 'UN'

        return result
