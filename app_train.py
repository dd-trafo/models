from corenlp import Cleaner, SentenceClassifier, SentenceSplitter, Paraphraser

splitter = SentenceSplitter(language='EN')
cleaner = Cleaner()

path = '/tmp/bert-base-cased/'
model = SentenceClassifier.from_pretrained(path)

path_train = '/opt/dev/corenlp/sentences/train'
df = model.load_sentences(path_train, clean=True)

model.load_data(df=df, test_size=0.05)

model.train(
    n_epoch=20,
    batch_size=12,  #16,
    learning_rate=3e-6,
    patience=2,
    path_output='/tmp/trained_model',
)
