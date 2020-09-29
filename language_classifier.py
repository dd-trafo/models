import pycld2 as cld2
import cld3
import langdetect


class LanguageClassifier:
    def __init__(self,
                 languages=None,
                 methods: list = ['cld2', 'cld3', 'langdetect']) -> None:

        from ..cleaner.cleaner import Cleaner

        if languages is None:
            self.languages = languages
        else:
            self.languages = [language.upper() for language in languages]

        self.methods = methods
        self.cleaner = Cleaner()

    def detect(self, txt: str) -> str:
        txt = self.cleaner.clean(txt)

        outputs = []
        for method in self.methods:
            try:
                if method == 'cld2':
                    is_reliable, text_bytes_found, details = cld2.detect(txt)

                    results = [result[1].upper() for result in details]
                elif method == 'cld3':
                    results = [
                        result.language.upper()
                        for result in cld3.get_frequent_languages(txt,
                                                                  num_langs=5)
                    ]
                elif method == 'langdetect':
                    results = [
                        result.lang.upper()
                        for result in langdetect.detect_langs(txt)
                    ]
                else:
                    raise NotImplementedError(
                        f'Method {method} not implemented.')

                if self.languages is not None:
                    results = [
                        result for result in results
                        if result in self.languages
                    ]

                if len(results) > 0:
                    outputs.append(results[0])
                else:
                    outputs.append('UN')
            except Exception as e:
                print(f'Error: {e}')
                outputs.append('UN')

        outputs = [output for output in outputs if output != 'UN']
        if len(outputs) > 0:
            return max(set(outputs), key=outputs.count)
        else:
            return 'UN'
