import hashlib
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from time import sleep

import numpy as np
from corenlp import Cleaner, Paraphraser, SentenceClassifier, TimeIt
from eventlet.queue import Queue
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

namespace = '/test'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretphrase'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

#async_mode = 'threading'
async_mode = 'eventlet'
socketio = SocketIO(app, async_mode=async_mode)

tasks = Queue()
results = Queue()

path_annotation = Path('/tmp/annotation')


class Director:
    def __init__(self, max_num_candidates=20, batch_size=8):
        print('Director: init()')

        self.max_num_candidates = max_num_candidates
        self.batch_size = batch_size

        self.cleaner = Cleaner()

        # Loading universe
        with TimeIt('Loading universe'):
            universe_file = Path(
                '.../universe.txt')
            with open(universe_file, 'r', encoding='utf-8') as handle:
                self.universe = handle.read().splitlines()

            self.universe = [self.cleaner.clean(u) for u in self.universe]

            random.shuffle(self.universe)

        self.candidates_universe = list()
        self.candidates_paraphrased = list()

        # Loading sentence classifier
        with TimeIt('Loading sentence classifier'):
            path = '.../model_v1'
            self.model = SentenceClassifier.from_pretrained(path)

            # Load existing sentences
            path_train = '.../train'
            df = self.model.load_sentences(path_train, clean=True)
            self.dataset = df['SENTENCE'].to_list()
            random.shuffle(self.dataset)

        with TimeIt('Loading paraphraser'):
            self.para = Paraphraser(type='generation', path='.../bart-base')

    def search(self):
        if (np.random.uniform() > 0.1) and (len(self.candidates_universe) <
                                            self.max_num_candidates):
            self.evaluate_universe()
        elif (len(self.candidates_paraphrased) < self.max_num_candidates):
            self.evaluate_paraphraser()
        else:
            return

    def evaluate_universe(self):
        print('Director: evaluate_universe()')

        if len(self.universe) == 0:
            print('Universe depleted. Returning...')
            return

        txts = list()
        for i in range(self.batch_size):
            txts.append(self.universe.pop())

        for pred in self.model.predict(txts):
            probability = pred['LABELS']['SUSTAINABLE']

            if abs(probability - 0.5) <= 0.4:
                self.candidates_universe.append({
                    'SENTENCE': pred['SENTENCE'],
                    'PROBABILITY': probability
                })

        self.candidates_universe = sorted(
            self.candidates_universe,
            key=lambda x: abs(x['PROBABILITY'] - 0.5),
            reverse=True)

    def evaluate_paraphraser(self):
        print('Director: evaluate_paraphraser()')

        if len(self.dataset) == 0:
            print('Dataset depleted. Returning...')
            return

        sentence = self.dataset.pop()

        explanation = self.model.explain(sentence, n_max=128, n_expand=100)

        paraphrased_sentences = self.para.adaptive_paraphrase(
            explanation,
            weight_threshold=0.0,
            do_sample=True,
            num_return_sequences=3,
            early_stopping=True,
        )

        self.candidates_paraphrased.extend(paraphrased_sentences)

    def pick(self):
        print(f'Director: pick()')

        choices = list()
        p = list()
        if len(self.candidates_universe):
            choices.append('universe')
            p.append(1.0)

        if len(self.candidates_paraphrased):
            choices.append('paraphrased')
            p.append(0.5)

        if len(choices) == 0:
            return None

        p = np.array(p)
        p = p / np.sum(p)

        choice = np.random.choice(choices, 1, p=p)[0]

        print(f'choice = "{choice}"')

        if choice == 'universe':
            return self.candidates_universe.pop(0)['SENTENCE']
        elif choice == 'paraphrased':
            return self.candidates_paraphrased.pop(0)
        else:
            return None


@app.route('/')
def index():
    return render_template('index.html', async_mode=async_mode)


@socketio.on('connect', namespace=namespace)
def connect():
    print('Client connected', request.sid)


@socketio.on('generate', namespace=namespace)
def generate(num_requests):
    global tasks

    print(f'generate(num_requests={num_requests})')
    for i in range(num_requests):
        tasks.put(None)


@socketio.on('disconnect', namespace=namespace)
def disconnect():
    print('Client disconnected', request.sid)


def get_hash(string: str) -> str:
    return hashlib.sha1(string.encode('utf-8')).hexdigest()


def get_timestamp() -> str:
    return str(int(time.time() * 1000))


def save_feedback(path: Path, sentence: str, labels: str) -> None:
    os.makedirs(path, exist_ok=True)
    filename = path / f'{get_hash(sentence)}_{get_timestamp()}.json'
    annotation = {'SENTENCE': sentence, 'LABELS': labels}
    with open(filename, 'w', encoding='utf-8') as handle:
        json.dump(annotation, handle)


@socketio.on('annotate', namespace=namespace)
def annotate(data):

    sentence = data['sentence']
    labels = data['labels']
    print(f'sentence = {sentence}, labels = {labels}')

    if labels != 'SKIP':
        save_feedback(path_annotation, sentence, labels)

    tasks.put(None)


@socketio.on('analyze', namespace=namespace)
def analyze(json):
    global tasks

    print(f'analyze: {json}')
    task = str(json)

    tasks.put(task)


def queue_watcher():
    print('Starting queue_watcher')

    global results

    while True:
        socketio.sleep(0.5)
        if not results.empty():
            result = results.get()
            print(f'result = {result}')
            #results.task_done()

            socketio.emit('sentence', result, namespace=namespace)
            # eventlet.sleep()

            print('queue_watcher emitted output')


def model():
    global tasks
    global results

    print('Starting model')

    director = Director()

    while True:
        socketio.sleep(0.5)
        director.search()

        if not tasks.empty():
            sentence = director.pick()
            if sentence is not None:
                tasks.get()
                results.put(sentence)


if __name__ == '__main__':

    socketio.start_background_task(target=model)
    socketio.start_background_task(target=queue_watcher)
    socketio.run(app, host='127.0.0.1', port=5000, debug=False)
