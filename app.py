import hashlib
import json
import os
import re
import time
from pathlib import Path

from corenlp import Cleaner, SentenceClassifier, SentenceSplitter
from flask import Flask, render_template, request, send_file

# export FLASK_APP=app.py
# export FLASK_ENV=development

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

### load model
path = '/tmp/trained_model'
model = SentenceClassifier.from_pretrained(path, device='cuda')

labels = model.labels
labels = [label for label in labels if label != 'NEUTRAL']
labels_str = "'" + "','".join(labels) + "'"

splitter = SentenceSplitter(language='EN')
cleaner = Cleaner()

path_annotation = Path('/.../annotation')


def get_unique_elements(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def get_hash(string: str) -> str:
    return hashlib.sha1(string.encode('utf-8')).hexdigest()


def get_timestamp() -> str:
    #return datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    return str(int(time.time() * 1000))


def save_feedback(path: Path, sentence: str, labels: str) -> None:
    os.makedirs(path, exist_ok=True)
    filename = path / f'{get_hash(sentence)}_{get_timestamp()}.json'
    annotation = {'SENTENCE': sentence, 'LABELS': labels}
    with open(filename, 'w', encoding='utf-8') as handle:
        json.dump(annotation, handle)


@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    response = request.json
    txt = response['text']
    txt = cleaner.clean(txt)

    # txt = re.sub('<[^<]+?>', '', txt)

    #print(f'txt = {txt}')
    # print(f'len(txt) = {len(txt)}')

    if len(txt) == 0:
        output = dict()
    else:
        #sentences = splitter.split(txt)
        sentences = txt.split('\n')
        sentences = get_unique_elements(sentences)
        output = model.predict(sentences)

    return json.dumps(output)


@app.route('/save_feedback', methods=['POST'])
def save_feedback_():
    try:
        response = request.json
        sentence = response['sentence']
        labels = response['labels']

        print(f'sentence = "{sentence}", labels = {labels}')

        save_feedback(path_annotation, sentence, labels)

        return 'Saved.'
    except Exception as e:
        return str(e)


@app.route('/annotate')
def annotate():
    return render_template('annotate.html', labels=labels_str)


@app.route('/live')
def live():
    return render_template('live.html', labels=labels_str)


@app.route('/')
def index():
    return 'OK'
