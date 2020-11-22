import re

from unidecode import unidecode


class Cleaner:
    def __init__(
        self,
        characters_to_keep:
        str = 'äüöÄÜÖßàÀâÂèÈéÉêÊëËîÎïÏôÔùÙûÛÿŸçÇœŒÁÌÍÒÓÚáìíòóú°',
        remove_linebreaks: bool = False,
    ) -> None:

        self.characters_to_keep = ''.join(set(characters_to_keep))
        self.remove_linebreaks = remove_linebreaks

        # Create a dictionary containing each character to keep
        # and its replacement string
        self.forward_sub = dict()
        for i, char in enumerate(self.characters_to_keep):
            self.forward_sub[char] = f'___char{i:02d}___'

        # Swap key/value in forward_sub
        self.backward_sub = dict()
        for key in self.forward_sub.keys():
            self.backward_sub[self.forward_sub[key]] = key

        # Compile for best performance
        self.forward_regex = re.compile('|'.join(
            map(re.escape, sorted(self.forward_sub, key=len, reverse=True))))

        self.backward_regex = re.compile('|'.join(
            map(re.escape, sorted(self.backward_sub, key=len, reverse=True))))

    def clean(self, txt: str) -> str:
        # Make sure it's a string
        if type(txt) != str:
            txt = str(txt)

        # Remove any xml/html tags
        txt = re.sub('<[^<]+?>', '', txt)

        # Replace typographic quotes
        txt = re.sub('[«»]', '"', txt)

        # Replace all non-ascii characters to keep
        txt = self.forward_regex.sub(
            lambda match: self.forward_sub[match.group(0)], txt)

        # Perform transliteration by unidecode
        try:
            txt = unidecode(txt)
        except Exception as e:
            print(e)

        # Translate special characters back
        txt = self.backward_regex.sub(
            lambda match: self.backward_sub[match.group(0)], txt)

        if self.remove_linebreaks:
            # Replace any newlines
            txt = re.sub('[\n\r]', ' ', txt)

        # Remove multi consecutive whitespace
        txt = re.sub(' +', ' ', txt)

        # Replace double commas and double single quotes
        txt = re.sub(',,(?!,)', '"', txt)
        txt = re.sub("''(?!')", '"', txt)

        # Remove superfluous whitespace
        txt = re.sub(r' +([,.%:\)\]])', r'\g<1>', txt)
        txt = re.sub(r'([\(\[]) +', r'\g<1>', txt)
        txt = re.sub(r'\( +', '(', txt)

        # Remove leading or trailing whitespace
        txt = txt.strip()

        return txt
