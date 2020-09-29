import numpy as np
import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
import torch
from tqdm import tqdm, trange
import os
from sklearn.preprocessing import MultiLabelBinarizer
import glob
import ntpath
import pickle
import hashlib
import datetime
import random
import re
import time
from pathlib import Path


def sanitize_label(label: str) -> str:
    """
    Sanitize the label (string) from non-ASCII characters and whitespaces.
    """
    initial_label = label
    label = re.sub('[^A-Za-z0-9_ ]', '', label)
    label = re.sub(' +', ' ', label)
    label = re.sub(' ', '_', label)

    if label != initial_label:
        print(f'Sanitized label from "{initial_label}" to "{label}"')

    return label


def load_sentences(path, clean: bool = True) -> pd.DataFrame:
    """
    Load sentences into a dataframe.
    - LABELNAME.txt: each line is a sentence assigned to label LABELNAME
    - 12345.json: contains one (1) sentence and its corresponding labels
    """

    df_list = list()

    # Read txt files and extract each line as a sentence
    path = Path(path)
    files_txt = path.rglob('*.txt')

    for file in tqdm(files_txt):
        label = sanitize_label(ntpath.basename(file.with_suffix('')))

        sentences = []
        with open(file, encoding='utf-8') as handle:
            sentences = handle.readlines()
            sentences = [
                re.sub('[\n\r]', '', sentence) for sentence in sentences
            ]

        df_list.append(
            pd.DataFrame(data={
                'SENTENCE': sentences,
                'LABELS': label
            }))

    # Read individual sentences from json
    files_json = list(path.rglob('*.json'))

    # Determine duplicates sentences and pick the latest version
    df_json = pd.DataFrame(data={'FILES': files_json})

    df_json['FILES'] = df_json['FILES'].apply(
        lambda file: ntpath.basename(file))

    df_json[['HASH', 'DATE_TIME']] = df_json.FILES.str.extract(
        '([a-z0-9]+)_(\d{4}-\d{2}-\d{2}_\d{6}).json')

    df_json = df_json.sort_values(['HASH', 'DATE_TIME'], ascending=False)

    # For each sentence, keep the latest version only
    df_json = df_json.drop_duplicates(subset=['HASH'], keep='first')

    files_json = df_json['FILES'].tolist()

    series_list = list()
    for file in tqdm(files_json):
        series_list.append(pd.read_json(path / file, typ='series'))

    df_list.append(pd.DataFrame(series_list))

    df = pd.concat(df_list, axis=0).reset_index(drop=True)

    df['LABELS'] = df['LABELS'].fillna('NEUTRAL')

    # empty sentences should be labelled 'NEUTRAL'
    df.loc[df['SENTENCE'].isna(), 'LABELS'] = 'NEUTRAL'
    df['SENTENCE'] = df['SENTENCE'].fillna('')

    if clean:
        from ..cleaner.cleaner import Cleaner
        cleaner = Cleaner()
        df['SENTENCE'] = df['SENTENCE'].apply(
            lambda sentence: cleaner.clean(sentence))

        del cleaner

    df = df.drop_duplicates().reset_index(drop=True)

    return df


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    BERT model for multi-label setup based-off of BertForSequenceClassification.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size,
                                          self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, ) + outputs[
            2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            outputs = (loss, ) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def read_txt(file):
    with open(file) as f:
        lines = f.read().splitlines()

    return lines


def evaluate_universe(universe: pd.Series,
                      batch_size: int = 4,
                      n_return: int = 4,
                      threshold: float = 0.1,
                      max_time_sec: int = 10):

    if n_return < 1:
        raise Exception('n_return must be >= 1.')

    n = len(universe)
    if n < 1:
        raise Exception('Universe does not contain any elements.')

    random_indices = np.random.choice(n, size=n, replace=False)

    n_batch = np.ceil(n / batch_size).astype('int')

    start_time = time.time()

    counter = 0

    candidates = pd.DataFrame(columns=['SENTENCE', 'PROB'])

    counter_candidates = 0

    model.eval()
    with torch.no_grad():
        #pbar = tqdm(gen, total=n_return)
        with trange(n_batch) as pbar:
            for _ in pbar:
                pbar.set_description(
                    f'Candidates from universe ({counter_candidates}/{n_return})'
                )

                batch_indices = random_indices[counter:(counter + batch_size)]
                counter = counter + batch_size

                batch = universe.iloc[batch_indices]

                inputs = tokenizer(batch.to_list(),
                                   padding=True,
                                   return_tensors='pt')

                if device == 'cuda':
                    for key in inputs.keys():
                        inputs[key] = inputs[key].to(device)

                outputs = model(**inputs)

                predictions = outputs[0].sigmoid().detach().cpu().numpy()

                for i, prediction in enumerate(predictions):
                    sample = pd.Series(
                        name=universe.index[batch_indices[i]],
                        data={
                            'SENTENCE': batch.iloc[i],
                            'PROB': prediction[0]
                            #'LABELS': dict(zip(all_labels, pred))
                        })

                    candidates = candidates.append(sample)

                    if abs(0.5 - prediction[0]) <= threshold:
                        counter_candidates = counter_candidates + 1

                if counter_candidates >= n_return:
                    #print('Found n_return items.')
                    pbar.set_description(
                        f'Candidates from universe ({counter_candidates}/{n_return})'
                    )
                    break

                if (time.time() - start_time) >= max_time_sec:
                    print('max_time_sec exceeded.')
                    pbar.set_description(
                        f'Candidates from universe ({counter_candidates}/{n_return})'
                    )
                    break

    candidates = candidates.sort_values(by='PROB', key=lambda x: abs(0.5 - x))

    return candidates.head(n_return)


path_sentences = Path('/opt/dev/corenlp/')

label = 'SUSTAINABILITY'
x = read_txt(path_sentences / f'{label}.txt')
df1 = pd.DataFrame(x, columns=['SENTENCE'])
df1['LABELS'] = label

label = 'NEUTRAL'
x = read_txt(path_sentences / f'{label}.txt')
df2 = pd.DataFrame(x, columns=['SENTENCE'])
df2['LABELS'] = label

df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
df['P'] = 1.0

label = 'UNIVERSE'
universe_list = read_txt(path_sentences / f'{label}.txt')
universe = pd.Series(universe_list)

batch_size = 8

df_train = df.copy()

train_sampler = Sampler(df_train, balance=True, use_weights=True)

all_labels = train_sampler.get_labels()

n_labels = len(all_labels)

label_encoder = MultiLabelBinarizer()
label_encoder.fit([all_labels])

now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

path_model = '/tmp/bert-base-german-dbmdz-cased/'
path_model = Path(path_model)

# Load model
tokenizer = BertTokenizer.from_pretrained(str(path_model))
model = BertForMultiLabelSequenceClassification.from_pretrained(
    str(path_model), num_labels=n_labels)

weight_decay = 0
warmup_ratio = 0.06
learning_rate = 4e-5
adam_epsilon = 1e-8
#device = 'cpu'
max_grad_norm = 1.0

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [{
    'params': [
        p for n, p in model.named_parameters()
        if not any(nd in n for nd in no_decay)
    ],
    'weight_decay':
    weight_decay,
}, {
    'params': [
        p for n, p in model.named_parameters()
        if any(nd in n for nd in no_decay)
    ],
    'weight_decay':
    0.0,
}]

#n_batch = np.ceil(len(df_train) / batch_size).astype(int)
n_batch = 8

### train

n_epoch = 30

t_total = batch_size * n_batch * n_epoch * 2

warmup_steps = int(np.ceil(t_total * warmup_ratio))

optimizer = AdamW(optimizer_grouped_parameters,
                  lr=learning_rate,
                  eps=adam_epsilon)

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=t_total)

# Set model to training mode
model.zero_grad()
model.train()

if torch.cuda.is_available():
    device = 'cuda'
    model.to(device)
else:
    device = 'cpu'

a = np.exp(-np.log(10.0) / n_batch)

for i_epoch in range(n_epoch):

    loss_train = 0.0
    i_batch = 0

    train_sampler = Sampler(df_train, balance=True, use_weights=True)

    gen = train_sampler.generator(batch_size=batch_size)

    pbar = tqdm(gen, total=n_batch)
    for sentences, labels in pbar:

        # Tokenize sentences from train_sampler
        inputs = tokenizer(sentences, padding=True, return_tensors='pt')

        # Add one-hot-encoded labels as tensor
        inputs['labels'] = torch.tensor(
            label_encoder.transform([tuple(x) for x in labels]))

        if device == 'cuda':
            for key in inputs.keys():
                inputs[key] = inputs[key].to(device)

        # Forward pass
        outputs = model(**inputs)

        loss = outputs[0]

        current_loss = loss.item()
        loss_train = loss_train + current_loss
        average_loss = loss_train / (i_batch + 1)

        pbar.set_description(
            f'Epoch {i_epoch + 1}, train loss: {current_loss:.5f} (avg: {average_loss:.5f})'
        )

        # Backpropagation
        loss.backward()

        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Update learning rate schedule
        scheduler.step()
        model.zero_grad()

        i_batch = i_batch + 1

        train_sampler.modify_weights(lambda p: max(a * p, 1.0))

        if i_batch > n_batch:
            break

    candidates = evaluate_universe(universe,
                                   max_time_sec=20,
                                   batch_size=batch_size,
                                   n_return=batch_size,
                                   threshold=0.2)

    candidates = candidates.drop(columns=['PROB'])

    candidates['LABELS'] = candidates['SENTENCE'].apply(
        lambda x: 'SUSTAINABILITY' if bool(
            re.search(
                r'umwelt|\bwald|\berde|\bglobali|\bmüll|\bplanet|verschmutz|\bwäld',
                x,
                flags=re.IGNORECASE)) else 'NEUTRAL')

    print(candidates)

    candidates['P'] = 10.0

    df_train['P'] = train_sampler.df_p

    df_train = df_train.append(candidates, ignore_index=True)

    universe = universe.drop(candidates.index)


def evaluate(sentences, batch_size=4):
    n = len(sentences)
    counter = 0

    indices = list(range(n))

    n_batch = np.ceil(n / batch_size).astype('int')

    output = list()

    model.eval()
    with torch.no_grad():
        for _ in trange(n_batch):
            batch = sentences[counter:(counter + batch_size)]
            counter = counter + batch_size

            inputs = tokenizer(batch, padding=True, return_tensors='pt')

            if device == 'cuda':
                for key in inputs.keys():
                    inputs[key] = inputs[key].to(device)

            # Evaluate model
            outputs = model(**inputs)

            preds = outputs[0].sigmoid().detach().cpu().numpy()

            for i, pred in enumerate(preds):
                output.append({
                    'SENTENCE': sentences[i],
                    'LABELS': dict(zip(all_labels, pred))
                })

    return output


sentences = universe[0:3]
sentences = ['Es gibt viele Atome in der Physik.']
sentences = ['Ich möchte heiraten.']
sentences = ['Es geht um Umweltfragen.']
sentences = ['Der Planet ist blau.']

evaluate(sentences)

output = evaluate(universe[0:5])

evaluate_universe(universe)

evaluate_universe(universe, max_time_sec=60, batch_size=8, n_return=8)
