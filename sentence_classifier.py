import datetime
import hashlib
import os
import pickle
import platform
from typing import List, Union

import cachetools
import numpy as np
from pathlib import Path
import pandas as pd
import random
import re
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
import torch
from tqdm import tqdm, trange

from sampler import Sampler
from cacher import Cacher


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


def load_sentences(path: str, clean: bool = True) -> pd.DataFrame:
    """
    Load sentences into a Pandas dataframe.
    - LABELNAME.txt: each line is a sentence assigned to label LABELNAME
    - 12345.json: contains one (1) sentence and its corresponding labels
    """

    df_list = list()

    # Read txt files and extract each line as a sentence
    path = Path(path)
    files_txt = list(path.rglob('*.txt'))

    for file_txt in tqdm(files_txt):
        label = sanitize_label(file_txt.stem)

        sentences = []
        with open(file_txt, encoding='utf-8') as handle:
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

    df_json['FILES'] = df_json['FILES'].apply(lambda file: file.name)

    df_json[['HASH', 'DATE_TIME']] = df_json.FILES.str.extract(
        r'([a-z0-9]+)_(\d{4}-\d{2}-\d{2}_\d{6}).json')

    df_json = df_json.sort_values(['HASH', 'DATE_TIME'], ascending=False)

    # For each sentence, keep the latest version only
    df_json = df_json.drop_duplicates(subset=['HASH'], keep='first')

    files_json = df_json['FILES'].tolist()

    series_list = list()
    for file_json in tqdm(files_json):
        series_list.append(pd.read_json(path / file_json, typ='series'))

    df_list.append(pd.DataFrame(series_list))

    df = pd.concat(df_list, axis=0).reset_index(drop=True)

    df['LABELS'] = df['LABELS'].fillna('NEUTRAL')

    # empty sentences should be labelled 'NEUTRAL'
    df.loc[df['SENTENCE'].isna(), 'LABELS'] = 'NEUTRAL'
    df['SENTENCE'] = df['SENTENCE'].fillna('')

    if clean:
        from cleaner import Cleaner
        cleaner = Cleaner()
        df['SENTENCE'] = df['SENTENCE'].apply(cleaner.clean)

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


# pylint: disable=not-callable
class SentenceClassifier:
    def __init__(
            self,
            df_train: pd.DataFrame,
            df_eval: pd.DataFrame,
            neutral_label: str = 'NEUTRAL',
            device: str = 'cpu',
            path_model:
        str = '/mnt/david_tmp/models/DE/bert-base-german-dbmdz-cased/',
            path_trained_model: str = '/tmp',
            path_cache: str = '/tmp/cache') -> None:

        #self.neutral_label = neutral_label

        self.device = device

        self.train_sampler = Sampler(df_train,
                                     neutral_label=neutral_label,
                                     balance=True)

        self.eval_sampler = Sampler(df_eval,
                                    neutral_label=neutral_label,
                                    balance=False)

        self.path_trained_model = Path(path_trained_model)

        self.labels = self.train_sampler.get_labels()
        self.n_labels = len(self.labels)

        self.label_encoder = MultiLabelBinarizer()
        self.label_encoder.fit([self.labels])

        self.path_model = Path(path_model)

        self.now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.path_trained_model = self.path_trained_model / self.now

        #self.model_hash = self.create_random_hash()
        self.model_hash = 'm123'

        self.cacher = None
        self.path_cache = path_cache

        # Load model
        self.tokenizer = BertTokenizer.from_pretrained(str(self.path_model))
        self.model = BertForMultiLabelSequenceClassification.from_pretrained(
            str(self.path_model), num_labels=self.n_labels)

    def train(
        self,
        n_epoch: int,
        batch_size: int,
    ) -> None:

        weight_decay = 0
        warmup_ratio = 0.06
        learning_rate = 4e-5
        adam_epsilon = 1e-8
        device = 'cpu'
        max_grad_norm = 1.0
        #save_model_every_epoch = True

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in self.model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            weight_decay,
        }, {
            'params': [
                p for n, p in self.model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.0,
        }]

        n_batch = self.train_sampler.get_n_batch(batch_size)
        n_batch_eval = self.eval_sampler.get_n_batch(batch_size)

        t_total = batch_size * n_batch * n_epoch

        warmup_steps = np.ceil(t_total * warmup_ratio).astype(int)

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=learning_rate,
                          eps=adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_total)

        # Set model to training mode
        self.model.zero_grad()
        self.model.train()

        # if device == 'cuda':
        #     self.model.cuda()
        #     self.model.to(device)

        for i_epoch in range(n_epoch):

            loss_train = 0.0
            i_batch = 0

            pbar = tqdm(self.train_sampler.generator(batch_size=batch_size),
                        total=n_batch)
            for sentences, labels in pbar:

                # Tokenize sentences from train_sampler
                inputs = self.tokenizer(sentences,
                                        padding=True,
                                        return_tensors='pt')

                # Add one-hot-encoded labels as tensor
                inputs['labels'] = torch.tensor(
                    self.label_encoder.transform([tuple(x) for x in labels]))

                # Forward pass
                outputs = self.model(**inputs)

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_grad_norm)

                #
                optimizer.step()

                # Update learning rate schedule
                scheduler.step()
                self.model.zero_grad()

                i_batch = i_batch + 1

            loss_eval = 0.0
            i_batch = 0
            # Put into evaluation mode
            self.model.eval()
            with torch.no_grad():
                pbar = tqdm(self.eval_sampler.generator(batch_size=batch_size),
                            total=n_batch_eval)
                for sentences, labels in pbar:

                    # Tokenize sentences from train_sampler
                    inputs = self.tokenizer(sentences,
                                            padding=True,
                                            return_tensors='pt')

                    # Add one-hot-encoded labels as tensor
                    inputs['labels'] = torch.tensor(
                        self.label_encoder.transform(
                            [tuple(x) for x in labels]))

                    outputs = self.model(**inputs)

                    #print(outputs)

                    current_loss = outputs[0].item()

                    loss_eval = loss_eval + current_loss

                    average_loss = loss_eval / (i_batch + 1)

                    pbar.set_description(
                        f'Epoch {i_epoch + 1}, eval loss:  {loss_eval:.5f} (avg: {average_loss:.5f})'
                    )

                    i_batch = i_batch + 1

            self.save(epoch=i_epoch + 1)

            #save()

    def save(self, epoch=None) -> None:
        """
        Save model, tokenizer and label_encoder to disk.
        """
        output_path = self.path_trained_model

        if epoch is not None:
            output_path = output_path / f'epoch_{epoch}'

        print(f'Saving to: {output_path}... ', end='')

        os.makedirs(output_path, exist_ok=True)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(output_path / 'label_encoder.pkl', 'wb') as handle:
            pickle.dump(self.label_encoder, handle)

        print('done')

    def get_hash(self, string: str) -> str:
        return hashlib.sha1(string.encode('utf-8')).hexdigest()

    def create_random_hash(self) -> str:
        try:
            hostname = platform.node()
        except:
            hostname = 'unknown'

        try:
            username = os.getlogin()
        except:
            username = 'unknown'

        s = hostname + username + str(datetime.datetime.now())

        return self.get_hash(s)

    # def retrieve_prediction(self, key: str) -> dict:
    #     with open(self.path_cache / f'{key}.pkl', 'rb') as handle:
    #         return pickle.load(handle)

    # def store_prediction(self, key: str, prediction: dict) -> None:
    #     with open(self.path_cache / f'{key}.pkl', 'wb') as handle:
    #         pickle.dump(prediction, handle)

    def predict(self,
                sentences: Union[str, List[str]],
                batch_size: int = 8,
                use_cache: bool = True,
                return_list: bool = True):

        if type(sentences) == str:
            sentences = [sentences]

        if len(sentences) == 0:
            raise Exception('No sentences supplied.')

        # df_INPUT contains the sentences in their original order
        df_INPUT = pd.DataFrame({'SENTENCE': sentences})
        df_INPUT['HASH'] = df_INPUT['SENTENCE'].apply(self.get_hash)

        # Identify unique sentences
        df_EVAL = df_INPUT.drop_duplicates(subset=['HASH']).reset_index(
            drop=True)

        if use_cache:
            # Initialize cacher
            if self.cacher is None:
                self.cacher = Cacher(path=self.path_cache,
                                     model_hash=self.model_hash)

            # Get previously computed labels from cacher
            # df_CACHED has columns ['HASH', 'LABELS']
            df_CACHED = self.cacher.get(hashes=df_EVAL['HASH'])

            # Let's determine what still needs to be evaluated
            # df_EVAL has columns ['SENTENCE', 'HASH', 'LABELS']
            df_EVAL = df_EVAL[~df_EVAL['HASH'].isin(df_CACHED['HASH'])]

            #df_unique_cached = df_unique.merge(df_cache, how='left', on='HASH')

            #df_unique = df_unique_cached[
            #    df_unique_cached['LABELS'].isna()].reset_index(drop=True)

        # Initialize column that stores predicted labels
        df_EVAL['LABELS'] = None

        # Determine number of sentences to evaluate the model with
        n = len(df_EVAL)

        # Given batch_size, we need to run n_batch iterations
        n_batch = np.ceil(n / batch_size).astype('int')

        counter = 0

        if n_batch > 0:
            self.model.eval()
            with torch.no_grad():
                for _ in trange(n_batch):

                    # Get indices of batch
                    batch_idx = df_EVAL.index[counter:(counter + batch_size)]

                    # These are the sentences, List[str]
                    batch = df_EVAL.loc[batch_idx, 'SENTENCE'].to_list()

                    counter = counter + 1

                    inputs = self.tokenizer(batch,
                                            padding=True,
                                            return_tensors='pt')

                    if self.device == 'cuda':
                        for key in inputs.keys():
                            inputs[key] = inputs[key].to(self.device)

                    outputs = self.model(**inputs)

                    predictions = outputs[0].sigmoid().detach().cpu().numpy()

                    for i, prediction in enumerate(predictions):
                        # SQLalchemy cannot JSON-serialize np.float's,
                        # hence, convert them to Python float
                        prediction = [value.item() for value in prediction]
                        prediction_labeled = dict(zip(self.labels, prediction))

                        df_EVAL['LABELS'][batch_idx[i]] = prediction_labeled

                        if use_cache:

                            self.cacher.cache({
                                'HASH': self.get_hash(batch[i]),
                                'LABELS': prediction_labeled
                            })

        df_EVAL = df_EVAL.drop(columns=['SENTENCE'])

        if use_cache:
            # Add cached predictions from above
            df_EVAL = pd.concat([df_EVAL, df_CACHED],
                                axis=0,
                                ignore_index=True)

        # Finally return the sentences in original order
        df_INPUT = df_INPUT.merge(df_EVAL, how='left', on='HASH')

        if return_list:
            return df_INPUT.apply(lambda row: {
                'SENTENCE': row['SENTENCE'],
                'LABELS': row['LABELS']
            },
                                  axis=1).to_list()
        else:
            return df_INPUT

        

if __name__ == "__main__":

    path_sentences = '.../sentences/'

    ### load data
    df = load_sentences(path_sentences, clean=False)
    df = df.iloc[0:5]

    batch_size = 6
    n_epoch = 1

    s = SentenceClassifier(df_train=df,
                           df_eval=df,
                           path_model='.../model_bert/')

    s.train(n_epoch=n_epoch, batch_size=batch_size)

    sentences = [
        'This is a sentence'
    ]

    s.predict(sentences, batch_size=8)
    s.predict(sentences, batch_size=8, use_cache=False)
