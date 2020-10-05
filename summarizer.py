from transformers import T5Tokenizer, T5ForConditionalGeneration


class Summarizer():
    def __init__(self, language='EN'):

        if language != 'EN':
            raise NotImplementedError('Summarizer only supports English.')

        path = '/tmp/t5-base'

        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = T5Tokenizer.from_pretrained(path)

    def summarize(self, txt):
        txt = txt.strip()

        inputs = self.tokenizer.encode(f'summarize: {txt}',
                                       return_tensors='pt')

        outputs = self.model.generate(inputs,
                                      num_beams=4,
                                      no_repeat_ngram_size=2,
                                      min_length=10,
                                      max_length=150,
                                      early_stopping=True,
                                      do_sample=False,
                                      length_penalty=1e-20)

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        summary = summary[0].upper() + summary[1:]

        return summary


txt = '''This is a longer text that should be summarized'''

smr = Summarizer()
summary = smr.summarize(txt)

print(summary)
