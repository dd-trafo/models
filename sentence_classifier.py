import datetime
import hashlib
import os
from typing import List, Union
from pathlib import Path
import pickle
import platform
import random
import re
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
import torch
from tqdm import tqdm, trange

from sampler import Sampler
from cacher import Cacher
from explainer import Explainer


class Timer():
    def __init__(self, message):
        print(f'{message}... ', end='')

    def __enter__(self):
        self.start = time.time()
        return None

    def __exit__(self, type, value, traceback):
        print(f'done [{time.time() - self.start:.2f}s]')


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    BERT model for a multi-label setup based-off of
    BertForSequenceClassification.
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
    def __init__(self,
                 path_model: str,
                 path_output: str = tempfile.gettempdir(),
                 device: Union[None, str] = None) -> None:

        self.path_model = Path(path_model)
        self.path_output = Path(path_output)
        self.now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.path_output = self.path_output / self.now

        self.train_sampler = None
        self.eval_sampler = None
        self.cacher = None

        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        # Check if we're loading an already fine-tuned model
        if os.path.isfile(self.path_model / 'label_encoder.pkl'):
            print('Found fine-tuned model.')

            # Load label_encoder
            with Timer('Loading label encoder'):
                with open(self.path_model / 'label_encoder.pkl',
                          'rb') as handle:
                    self.label_encoder = pickle.load(handle)

            # Load neutral label
            with Timer('Loading neutral label'):
                neutral_label_file = self.path_model / 'neutral_label.txt'
                if os.path.isfile(neutral_label_file):
                    with open(neutral_label_file, 'r') as handle:
                        self.neutral_label = handle.read()
                else:
                    self.neutral_label = 'NEUTRAL'

            self.labels = list(self.label_encoder.classes_)
            self.n_labels = len(self.labels)

            # Load model
            with Timer('Loading model'):
                self.tokenizer = BertTokenizer.from_pretrained(
                    str(self.path_model))
                self.model = BertForMultiLabelSequenceClassification.from_pretrained(
                    str(self.path_model), num_labels=self.n_labels)
        else:
            pytorch_file = self.path_model / 'pytorch_model.bin'
            if not os.path.isfile(pytorch_file):
                raise FileNotFoundError('Could not find `pytorch_model.bin`.')

            # We found a vanilla model. Don't load anything just yet, since we need to know
            # `n_labels` from load_data().
            print('Found vanilla model. Call `load_data(...)` to initialize.')
            self.neutral_label = None
            self.tokenizer = None
            self.model = None

    @classmethod
    def from_pretrained(cls, path):
        return SentenceClassifier(path)

    def load_data(self,
                  path: Union[None, str] = None,
                  df: Union[None, pd.DataFrame] = None,
                  df_eval: Union[None, pd.DataFrame] = None,
                  test_size: float = 0.3,
                  neutral_label: str = 'NEUTRAL') -> None:

        if (path is None) and (df is None):
            raise Exception('Supply either `path` or `df`.')

        if (path is not None) and (df is not None):
            print('`df` is ignored if both `path` and `df` are supplied.')

        if path is not None:
            with Timer('Loading sentences'):
                df = self.load_sentences(path)

        if df_eval is None:
            with Timer('Splitting into train and eval sets'):
                df_train, df_eval = train_test_split(df, test_size=test_size)
        else:
            df_train = df.copy()

        del df

        print(f'Sample size: train: {len(df_train)}, eval: {len(df_eval)}')

        self.neutral_label = neutral_label
        print(f'Neutral label is "{self.neutral_label}".')

        # Initialize samplers
        with Timer('Initializing samplers'):
            self.train_sampler = Sampler(df_train,
                                         neutral_label=neutral_label,
                                         balance=True)
            self.eval_sampler = Sampler(df_eval,
                                        neutral_label=neutral_label,
                                        balance=False)

        # If fine-tuned model has not been loaded
        if self.model is None:
            self.labels = self.train_sampler.get_labels()
            self.n_labels = len(self.labels)
            print(f'Found {self.n_labels} labels: {self.labels}')

            # Initialize label encoder
            with Timer('Initializing label encoder'):
                self.label_encoder = MultiLabelBinarizer()
                self.label_encoder.fit([self.labels])

            # Load model
            with Timer('Loading model'):
                self.tokenizer = BertTokenizer.from_pretrained(
                    str(self.path_model))
                self.model = BertForMultiLabelSequenceClassification.from_pretrained(
                    str(self.path_model), num_labels=self.n_labels)

    def _sanitize_label(self, label: str) -> str:
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

    def load_sentences(self, path: str, clean: bool = True) -> pd.DataFrame:
        """
        Load sentences into a Pandas dataframe.
        - LABELNAME.txt: each line is a sentence assigned to label LABELNAME
        - 12345.json: contains one (1) sentence and its corresponding labels
        """

        df_list = list()

        # Read txt files and extract each line as a sentence
        path = Path(path)
        files_txt = list(path.rglob('*.txt'))

        if len(files_txt) > 0:
            for file_txt in files_txt:
                label = self._sanitize_label(file_txt.stem)

                sentences = []
                with open(file_txt, encoding='utf-8') as handle:
                    sentences = handle.readlines()
                    sentences = [
                        re.sub('[\n\r]', '', sentence)
                        for sentence in sentences
                    ]

                df_list.append(
                    pd.DataFrame(data={
                        'SENTENCE': sentences,
                        'LABELS': label
                    }))

        # Read individual sentences from json
        files_json = list(path.rglob('*.json'))

        if len(files_json) > 0:
            # Determine duplicates sentences and pick the latest version
            df_json = pd.DataFrame(data={'FILES': files_json})

            df_json['FILES'] = df_json['FILES'].apply(lambda file: file.name)

            df_json[['HASH', 'DATE_TIME']] = df_json.FILES.str.extract(
                r'([a-z0-9]+)_(\d{4}-\d{2}-\d{2}_\d{6}).json')

            df_json = df_json.sort_values(['HASH', 'DATE_TIME'],
                                          ascending=False)

            # For each sentence, keep the latest version only
            df_json = df_json.drop_duplicates(subset=['HASH'], keep='first')

            files_json = df_json['FILES'].tolist()

            series_list = list()
            for file_json in files_json:
                series_list.append(pd.read_json(path / file_json,
                                                typ='series'))

            df_list.append(pd.DataFrame(series_list))

        # Check if we found something (either txt or json)
        assert len(df_list) > 0, f'No files found in {path}.'

        df = pd.concat(df_list, axis=0).reset_index(drop=True)

        df['LABELS'] = df['LABELS'].fillna('NEUTRAL')

        # Empty sentences should be labelled 'NEUTRAL'
        df.loc[df['SENTENCE'].isna(), 'LABELS'] = 'NEUTRAL'
        df['SENTENCE'] = df['SENTENCE'].fillna('')

        if clean:
            from cleaner import Cleaner
            cleaner = Cleaner()
            df['SENTENCE'] = df['SENTENCE'].apply(cleaner.clean)

            del cleaner

        df = df.drop_duplicates().reset_index(drop=True)

        return df

    def train(self,
              n_epoch: int,
              batch_size: int,
              device: Union[None, str] = None) -> None:

        if (self.train_sampler is None) or (self.eval_sampler is None):
            raise Exception('No data found (did you call `load_data(...)`?)')

        if device is not None:
            self.device = device

        # When modifying the model weights, invalidate the cacher and point to
        # a temp directory
        self.path_model = Path(
            tempfile.gettempdir()) / f'cacher_{s.create_random_hash()}'
        self.cacher = None

        # parameters
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

            with tqdm(self.train_sampler.generator(batch_size=batch_size),
                      total=n_batch) as pbar:

                for sentences, labels in pbar:

                    # Tokenize sentences from train_sampler
                    inputs = self.tokenizer(sentences,
                                            padding=True,
                                            return_tensors='pt')

                    # Add one-hot-encoded labels as tensor
                    inputs['labels'] = torch.tensor(
                        self.label_encoder.transform(
                            [tuple(x) for x in labels]))

                    # Forward pass
                    outputs = self.model(**inputs)

                    loss = outputs[0]

                    current_loss = loss.item()
                    loss_train = loss_train + current_loss
                    average_loss = loss_train / (i_batch + 1)

                    pbar.set_description(
                        f'Epoch {i_epoch + 1}, train: {current_loss:.5f} (avg: {average_loss:.5f})'
                    )

                    # Backpropagation of loss
                    loss.backward()

                    # Clip gradient norm
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   max_grad_norm)

                    # Update optimizer and scheduler
                    optimizer.step()
                    scheduler.step()

                    # Set gradients to zero
                    self.model.zero_grad()

                    i_batch = i_batch + 1

            loss_eval = 0.0
            i_batch = 0
            # Put model into evaluation mode
            self.model.eval()
            with torch.no_grad(), tqdm(
                    self.eval_sampler.generator(batch_size=batch_size),
                    total=n_batch_eval) as pbar:
                for sentences, labels in pbar:

                    # Tokenize sentences from eval_sampler
                    inputs = self.tokenizer(sentences,
                                            padding=True,
                                            return_tensors='pt')

                    # Add one-hot-encoded labels as tensor
                    inputs['labels'] = torch.tensor(
                        self.label_encoder.transform(
                            [tuple(x) for x in labels]))

                    outputs = self.model(**inputs)

                    current_loss = outputs[0].item()

                    loss_eval = loss_eval + current_loss

                    average_loss = loss_eval / (i_batch + 1)

                    pbar.set_description(
                        f'{len(f"Epoch {i_epoch + 1},") * " "} eval:  {loss_eval:.5f} (avg: {average_loss:.5f})'
                    )

                    i_batch = i_batch + 1

            #self.save(epoch=i_epoch)

    def save(self, epoch=None, path=None) -> None:
        """
        Save model, tokenizer and label_encoder to disk.
        """

        if path is None:
            path = self.path_output
        else:
            path = Path(path)

        if epoch is not None:
            path = path / f'epoch_{epoch}'

        with Timer(f'Saving to "{path}"'):
            os.makedirs(path, exist_ok=True)

            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

            with open(path / 'label_encoder.pkl', 'wb') as handle:
                pickle.dump(self.label_encoder, handle)

            with open(path / 'neutral_label.txt', 'w') as handle:
                handle.write(self.neutral_label)

        # Finally, update the path_model and reset the cacher
        self.path_model = path
        self.cacher = None

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
                return_list: bool = True,
                device: Union[None, str] = None):

        if type(sentences) == str:
            sentences = [sentences]

        if len(sentences) == 0:
            raise Exception('No sentences supplied.')

        if device is not None:
            self.device = device

        # df_INPUT stores sentences in their original order
        df_INPUT = pd.DataFrame({'SENTENCE': sentences})
        df_INPUT['HASH'] = df_INPUT['SENTENCE'].apply(self.get_hash)

        # Identify unique sentences
        df_EVAL = df_INPUT.drop_duplicates(subset=['HASH']).reset_index(
            drop=True)

        if use_cache:
            # Initialize cacher
            if self.cacher is None:
                self.cacher = Cacher(path=self.path_model)

            # Get previously computed labels from cacher
            # df_CACHED has columns ['HASH', 'LABELS']
            df_CACHED = self.cacher.get(HASHES=df_EVAL['HASH'])

            # Let's determine what still needs to be evaluated
            # df_EVAL has columns ['SENTENCE', 'HASH', 'LABELS']
            df_EVAL = df_EVAL[~df_EVAL['HASH'].isin(df_CACHED['HASH'])]

        # Initialize column that stores predicted labels
        df_EVAL['LABELS'] = None

        # Determine number of sentences to evaluate the model with
        n = len(df_EVAL)

        # Given batch_size, we need to run n_batch iterations
        n_batch = np.ceil(n / batch_size).astype(int)

        if n_batch > 0:
            self.model.eval()
            with torch.no_grad():
                counter = 0
                for _ in trange(n_batch):

                    # Get indices of batch
                    batch_idx = df_EVAL.index[counter:(counter + batch_size)]

                    # These are the sentences to be evaluated, List[str]
                    batch = df_EVAL.loc[batch_idx, 'SENTENCE'].to_list()

                    counter = counter + batch_size

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
                            self.cacher.set(HASH=self.get_hash(batch[i]),
                                            LABELS=prediction_labeled)

        df_EVAL = df_EVAL.drop(columns=['SENTENCE'])

        if use_cache:
            # Combine evaluation and cached predictions from above
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

    def explain(self,
                txt: str,
                label: str,
                min_removal: int = 1,
                max_removal: int = 3,
                use_auto_mode: bool = True,
                use_split: bool = False,
                use_cache: bool = True,
                n_expand: int = 10):

        if label not in self.labels:
            raise Exception(
                f'Selected label "{label}" is not among {self.labels}')

        exp = Explainer()
        explanation = exp.explain(txt=txt,
                                  tokenizer=self.tokenizer,
                                  predict=self.predict,
                                  label=label,
                                  min_removal=min_removal,
                                  max_removal=max_removal,
                                  use_auto_mode=use_auto_mode,
                                  use_split=use_split,
                                  use_cache=use_cache,
                                  n_expand=n_expand)

        return explanation


##################################
##################################
##################################

if __name__ == "__main__":

    path_model = '/tmp/bert-base-german-dbmdz-cased/'
    s = SentenceClassifier.from_pretrained(path_model)

if False:

    path_sentences = '/opt/python/env_huggingface/sentences/DE/'
    #df = s.load_sentences(path_sentences)
    s.load_data(path_sentences)

    n_epoch = 20
    batch_size = 8
    s.train(n_epoch=n_epoch, batch_size=batch_size)

    txt = 'Es geht um die Familie.'

    s.predict([txt, txt], use_cache=True)

    s.labels

    s.explain('This is a sentence.',
              label='ENVIRONMENT',
              use_auto_mode=True,
              use_cache=True)

    s.save()

    s.save(epoch=2)
