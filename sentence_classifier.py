import datetime
import hashlib
import os
import pickle
import platform
import random
import re
import tempfile
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm, trange
from transformers import (AdamW, BertForSequenceClassification, BertModel,
                          BertPreTrainedModel, BertTokenizer,
                          get_linear_schedule_with_warmup)
from transformers.modeling_outputs import SequenceClassifierOutput

from .cacher import Cacher
from .cleaner import Cleaner
from .explainer import Explainer
from .sampler import Sampler
from .time_it import TimeIt


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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))

        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# pylint: disable=not-callable
class SentenceClassifier:
    def __init__(
        self,
        path_model: str,
        exclusive_classes: bool = True,
        device: Union[None, str] = None,
    ) -> None:

        self.path_model = Path(path_model)

        self.exclusive_classes = exclusive_classes

        self.train_sampler = None
        self.eval_sampler = None
        self.cacher = None

        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'

        print(f'Using device "{self.device}"')

        label_encoder_file = self.path_model / 'label_encoder.pkl'

        # Check if we're loading an already fine-tuned model
        if not label_encoder_file.is_file():
            pytorch_file = self.path_model / 'pytorch_model.bin'
            if not pytorch_file.is_file():
                raise FileNotFoundError('Could not find `pytorch_model.bin`.')

            # We found a vanilla model. Don't load anything just yet, since we need to know
            # the number of labels from `load_data()``.
            print('Found vanilla model. Call `load_data(...)` to initialize.')
            self.neutral_label = None
            self.tokenizer = None
            self.model = None
        else:
            print('Found fine-tuned model.')

            # Load label_encoder
            with TimeIt('Loading label encoder'):
                with open(label_encoder_file, 'rb') as handle:
                    self.label_encoder = pickle.load(handle)

            # Check which type of encoder we have. This determines if classes
            # are exclusive or not.
            if type(self.label_encoder) == LabelEncoder:
                self.exclusive_classes = True
            elif type(self.label_encoder) == MultiLabelBinarizer:
                self.exclusive_classes = False
            else:
                raise Exception(
                    f'Cannot handle type of `label_encoder`: {type(self.label_encoder)}'
                )

            # Load neutral label
            with TimeIt('Loading neutral label'):
                neutral_label_file = self.path_model / 'neutral_label.txt'
                if neutral_label_file.is_file():
                    with open(neutral_label_file, 'r') as handle:
                        self.neutral_label = handle.read()
                else:
                    self.neutral_label = 'NEUTRAL'

            self.labels = list(self.label_encoder.classes_)
            self.n_labels = len(self.labels)

            # Load model
            with TimeIt('Loading model'):
                self.tokenizer = BertTokenizer.from_pretrained(
                    str(self.path_model))

                if self.exclusive_classes:
                    # LabelEncoder already knows about `neutral_label`, hence,
                    # use `n_labels` as it is
                    self.model = BertForSequenceClassification.from_pretrained(
                        str(self.path_model), num_labels=self.n_labels)
                else:
                    self.model = BertForMultiLabelSequenceClassification.from_pretrained(
                        str(self.path_model), num_labels=self.n_labels)

                if self.device == 'cuda':
                    self.model.to('cuda')

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return SentenceClassifier(*args, **kwargs)

    def load_data(
        self,
        path: Union[None, str] = None,
        df: Union[None, pd.DataFrame] = None,
        df_eval: Union[None, pd.DataFrame] = None,
        test_size: Union[None, float] = None,
        neutral_label: str = 'NEUTRAL',
        clean: bool = True,
    ) -> None:

        if (path is None) and (df is None):
            raise Exception('Supply either `path` or `df`.')

        if (path is not None) and (df is not None):
            print('`df` is ignored if both `path` and `df` are supplied.')

        if (df is not None) and (df_eval is not None) and (test_size
                                                           is not None):
            print(
                '`test_size` is ignored if both `df` and `df_eval` are supplied.'
            )

        if df is not None:
            assert 'SENTENCE' in df.columns, '`df` must have column `SENTENCE`'
            assert 'LABELS' in df.columns, '`df` must have column `LABELS`'

        if path is not None:
            with TimeIt('Loading sentences'):
                df = self.load_sentences(path, clean=clean)

        if self.exclusive_classes and df['LABELS'].str.contains(
                '|', regex=False).any():
            print('Conflicting sample(s):')
            print('----------------------')
            print(df[df['LABELS'].str.contains('|', regex=False)].head())
            raise Exception(
                'Option `exclusive_classes=True` requires one label per sentence.'
            )

        if df_eval is None:
            if test_size is None:
                test_size = 0.3

            with TimeIt('Splitting into train and eval sets'):
                df_train, df_eval = train_test_split(df,
                                                     test_size=test_size,
                                                     stratify=df['LABELS'])
        else:
            df_train = df.copy()

        del df

        print(f'Sample size: train: {len(df_train)}, eval: {len(df_eval)}')

        self.neutral_label = neutral_label
        print(f'Neutral label is "{self.neutral_label}".')

        # Initialize samplers
        with TimeIt('Initializing samplers'):
            self.train_sampler = Sampler(df_train,
                                         neutral_label=neutral_label,
                                         balance=True)
            self.eval_sampler = Sampler(df_eval,
                                        neutral_label=neutral_label,
                                        balance=False)

        # If fine-tuned model has not been loaded, the user requested a
        # vanilla model
        if self.model is None:
            self.labels = self.train_sampler.get_labels()
            self.n_labels = len(self.labels)
            print(f'Found {self.n_labels} label(s): {self.labels}')

            # Initialize label encoder
            with TimeIt('Initializing label encoder'):
                if self.exclusive_classes:
                    self.label_encoder = LabelEncoder()
                    # `LabelEncoder` must be fitted with `neutral_label`.
                    # Careful, it assigns numbers in alphabetic order!
                    self.label_encoder.fit([self.neutral_label] + self.labels)
                else:
                    # `MultiLabelBinarizer` outputs zeros for unknown labels
                    self.label_encoder = MultiLabelBinarizer()
                    self.label_encoder.fit([self.labels])

            # Load model
            with TimeIt('Loading model'):
                self.tokenizer = BertTokenizer.from_pretrained(
                    str(self.path_model))

                if self.exclusive_classes:
                    # We add one to `num_labels` to account for `neutral_label`
                    self.model = BertForSequenceClassification.from_pretrained(
                        str(self.path_model), num_labels=self.n_labels + 1)
                else:
                    self.model = BertForMultiLabelSequenceClassification.from_pretrained(
                        str(self.path_model), num_labels=self.n_labels)

                if self.device == 'cuda':
                    self.model.to('cuda')

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

            df_json['FILENAMES'] = df_json['FILES'].apply(
                lambda file: file.name)

            df_json[[
                'HASH', 'TIMESTAMP'
            ]] = df_json['FILENAMES'].str.extract(r'([a-z0-9]+)_(\d+).json')

            df_json = df_json.sort_values(['HASH', 'TIMESTAMP'],
                                          ascending=False)

            # For each sentence, keep the latest version only
            df_json = df_json.drop_duplicates(subset=['HASH'], keep='first')

            files_json = df_json['FILES'].tolist()

            series_list = list()
            for file_json in files_json:
                series_list.append(pd.read_json(file_json, typ='series'))

            df_list.append(pd.DataFrame(series_list))

        # Check if we found something (either txt or json)
        assert len(df_list) > 0, f'No files found in {path}.'

        df = pd.concat(df_list, axis=0).reset_index(drop=True)

        df['LABELS'] = df['LABELS'].fillna('NEUTRAL')
        df['LABELS'] = df['LABELS'].replace('', 'NEUTRAL')

        # Empty sentences should be labelled 'NEUTRAL'
        df.loc[df['SENTENCE'].isna(), 'LABELS'] = 'NEUTRAL'
        df['SENTENCE'] = df['SENTENCE'].fillna('')

        if clean:
            cleaner = Cleaner()
            df['SENTENCE'] = df['SENTENCE'].apply(cleaner.clean)

            del cleaner

        df = df.drop_duplicates().reset_index(drop=True)

        return df

    def train(
            self,
            n_epoch: int,
            batch_size: int,
            learning_rate: float = 4e-5,
            model_selection: str = 'min_eval',
            patience: int = -1,
            path_output: str = tempfile.gettempdir(),
    ) -> None:

        if (self.train_sampler is None) or (self.eval_sampler is None):
            raise Exception('No data found (did you call `load_data(...)`?)')

        # When modifying the model weights, invalidate the cacher and point to
        # a temp directory
        path_model = Path(
            tempfile.gettempdir()) / f'cacher_{self.create_random_hash()}'

        if path_output is None:
            now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
            path_output = Path(path_output) / now
        else:
            path_output = Path(path_output)

        self.cacher = None

        # Parameters
        weight_decay = 0
        warmup_ratio = 0.06
        adam_epsilon = 1e-8
        max_grad_norm = 1.0
        #save_model_every_epoch = True
        min_loss_eval = np.inf
        num_epochs_wo_improvement = 0

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

        # Get number of batches from sampler
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

                    # Add labels to `inputs`
                    if self.exclusive_classes:
                        # Flatten list of lists for `LabelEncoder`
                        labels = [label[0] for label in labels]

                        inputs['labels'] = torch.tensor(
                            self.label_encoder.transform(labels))
                    else:
                        inputs['labels'] = torch.tensor(
                            self.label_encoder.transform(
                                [tuple(label) for label in labels]))

                    if self.device == 'cuda':
                        for key in inputs.keys():
                            inputs[key] = inputs[key].to('cuda')

                    # Forward pass
                    outputs = self.model(**inputs)

                    loss = outputs[0]

                    current_loss = loss.item()
                    loss_train = loss_train + current_loss
                    average_loss = loss_train / (i_batch + 1)

                    pbar.set_description(
                        f'Epoch {i_epoch}, train: {current_loss:.5f} (avg: {average_loss:.5f})'
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

                    # Add labels to `inputs`
                    if self.exclusive_classes:
                        # Flatten list of lists for `LabelEncoder`
                        labels = [label[0] for label in labels]

                        inputs['labels'] = torch.tensor(
                            self.label_encoder.transform(labels))
                    else:
                        inputs['labels'] = torch.tensor(
                            self.label_encoder.transform(
                                [tuple(x) for x in labels]))

                    if self.device == 'cuda':
                        for key in inputs.keys():
                            inputs[key] = inputs[key].to('cuda')

                    outputs = self.model(**inputs)

                    loss = outputs[0]

                    current_loss_eval = loss.item()
                    loss_eval = loss_eval + current_loss_eval
                    average_loss_eval = loss_eval / (i_batch + 1)

                    pbar.set_description(
                        f'{len(f"Epoch {i_epoch},") * " "} eval:  {current_loss_eval:.5f} (avg: {average_loss_eval:.5f})'
                    )

                    i_batch = i_batch + 1

            if average_loss_eval < min_loss_eval:
                min_loss_eval = average_loss_eval
                if model_selection == 'min_eval':
                    self.save(path=path_output, quiet=True)
            else:
                num_epochs_wo_improvement += 1

            if (patience > 0) and (num_epochs_wo_improvement >= patience):
                print(
                    f'No improvement since {num_epochs_wo_improvement} epoch(s). Stopping.'
                )
                break
            #self.save(epoch=i_epoch)

    def save(
        self,
        epoch: Union[None, int] = None,
        path: Union[None, Path, str] = None,
        quiet: bool = False,
    ) -> None:
        """
        Save model, tokenizer and label_encoder to disk.
        """

        if (self.model is None) or (self.tokenizer is None):
            raise Exception('No model loaded. Did you call `load_data(...)`?')

        if path is None:
            path = tempfile.gettempdir()
            now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
            path = Path(path) / now
        else:
            path = Path(path)

        if epoch is not None:
            path = path / f'epoch_{epoch}'

        with TimeIt(f'Saving to "{path}"', quiet=quiet):
            os.makedirs(path, exist_ok=True)

            # Delete `cache.db` if it exists
            try:
                cache_file = path / 'cache.db'
                if cache_file.is_file():
                    cache_file.unlink()
            except Exception as e:
                print(f'Warning: {str(e)}')

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

    def predict(
        self,
        txt: Union[str, List[str]],
        batch_size: int = 8,
        use_cache: bool = True,
        return_dataframe: bool = False,
    ):

        if (self.model is None) or (self.tokenizer is None):
            raise Exception('No model loaded. Did you call `load_data(...)`?')

        if type(txt) == list:
            return_list = True
        else:
            return_list = False
            txt = [txt]

        assert len(txt) > 0, 'No txt supplied.'
        assert batch_size > 0, '`batch_size` must be a positive integer.'

        # df_INPUT stores txts in their original order
        df_INPUT = pd.DataFrame({'SENTENCE': txt})
        df_INPUT['HASH'] = df_INPUT['SENTENCE'].apply(self.get_hash)

        # Identify unique txts
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

        # Determine number of txts to evaluate the model with
        n = len(df_EVAL)

        # Given batch_size, we need to run n_batch iterations
        n_batch = np.ceil(n / batch_size).astype(int)

        if n_batch > 0:
            self.model.eval()
            with torch.no_grad():
                counter = 0
                for _ in range(n_batch):

                    # Get indices of batch
                    batch_idx = df_EVAL.index[counter:(counter + batch_size)]

                    # These are the txts to be evaluated, List[str]
                    batch = df_EVAL.loc[batch_idx, 'SENTENCE'].to_list()

                    counter = counter + batch_size

                    inputs = self.tokenizer(batch,
                                            padding=True,
                                            return_tensors='pt')

                    if self.device == 'cuda':
                        for key in inputs.keys():
                            inputs[key] = inputs[key].to('cuda')

                    outputs = self.model(**inputs)

                    if self.exclusive_classes:
                        # Exclusive classes requires normalization via softmax,
                        # s.t. it sums to one
                        predictions = outputs[0].softmax(
                            dim=1).detach().cpu().numpy()
                    else:
                        # For multi-label problems, each predicted dimension is
                        # independent, hence, requires sigmoid
                        predictions = outputs[0].sigmoid().detach().cpu(
                        ).numpy()

                    for i, prediction in enumerate(predictions):
                        # SQLalchemy cannot JSON-serialize np.float's,
                        # hence, convert them to Python float
                        prediction = [value.item() for value in prediction]

                        if self.exclusive_classes:
                            # Zip predictions with label_encoder's classes
                            prediction_labeled = dict(
                                zip(self.label_encoder.classes_, prediction))
                        else:
                            prediction_labeled = dict(
                                zip(self.labels, prediction))

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

        # Finally return the txts in original order
        df_INPUT = df_INPUT.merge(df_EVAL, how='left', on='HASH')

        if return_dataframe:
            return df_INPUT
        else:
            output = df_INPUT.apply(lambda row: {
                'SENTENCE': row['SENTENCE'],
                'LABELS': row['LABELS']
            },
                                    axis=1).to_list()

            if return_list:
                return output
            else:
                return output[0]

    def explain(
        self,
        txt: str,
        label: Union[None, str] = None,
        min_removal: int = 1,
        max_removal: int = 3,
        n_max: Union[None, int] = 96,
        test_size: Union[None, float] = 0.3,
        use_cache: bool = True,
        n_expand: int = 10,
    ):
        """
        Explain classification output by means of logistic regression.
        :param txt: Sentence/string to explain.
        :param label: Which label to explain, must be among the labels used
                      in training.
        """

        if (self.model is None) or (self.tokenizer is None):
            raise Exception('No model loaded. Did you call `load_data(...)`?')

        non_neutral_labels = [
            label for label in self.labels if label != self.neutral_label
        ]
        if (label is None) and len(non_neutral_labels) == 1:
            label = non_neutral_labels[0]

        if label not in self.labels:
            raise Exception(
                f'Selected label "{label}" is not among {self.labels}')

        exp = Explainer()
        return exp.explain(txt=txt,
                           tokenizer=self.tokenizer,
                           predict=self.predict,
                           label=label,
                           min_removal=min_removal,
                           max_removal=max_removal,
                           n_max=n_max,
                           test_size=test_size,
                           use_cache=use_cache,
                           n_expand=n_expand)

    def get_embedding(
        self,
        txt: Union[str, List[str]],
        batch_size: int = 8,
        device: Union[None, str] = None,
    ) -> Union[np.array, List[np.array]]:
        """
        Get model-internal representation of a supplied txt,
        i.e. its [CLS] embedding
        """

        if (self.model is None) or (self.tokenizer is None):
            raise Exception('No model loaded. Did you call `load_data(...)`?')

        if type(txt) == list:
            return_list = True
        else:
            return_list = False
            txt = [txt]

        if device is not None:
            self.device = device

        assert batch_size > 0, '`batch_size` must be a positive integer.'

        # Determine number of elements to evaluate the model with
        n = len(txt)

        # Given batch_size, we need to run n_batch iterations
        n_batch = np.ceil(n / batch_size).astype(int)

        cls_embeddings = list()
        if n_batch > 0:
            self.model.eval()
            with torch.no_grad():
                counter = 0
                for _ in trange(n_batch):
                    # Select a batch of strings
                    batch = txt[counter:(counter + batch_size)]
                    counter = counter + batch_size

                    # Tokenize
                    inputs = self.tokenizer(batch,
                                            padding=True,
                                            return_tensors='pt')

                    if self.device == 'cuda':
                        for key in inputs.keys():
                            inputs[key] = inputs[key].to(self.device)

                    # Forward pass of inputs to receive embeddings
                    outputs = self.model.forward(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                    # Pick from outputs:
                    # -1 -> the last layer
                    #  : -> all txts
                    #  0 -> [CLS] token
                    #  : -> all dimensions
                    embeddings = outputs.hidden_states[-1][:, 0, :].numpy()

                    # For all supplied txts
                    for i in range(embeddings.shape[0]):
                        cls_embeddings.append(embeddings[i])

        if return_list:
            # User supplied a list, hence, also return a list
            return cls_embeddings
        else:
            # User supplied a string, hence, return a single element
            return cls_embeddings[0]

    def step(
        self,
        sentences: List[str],
        labels: Union[List[str], List[List[str]]],
        learning_rate: float = 4e-5,
        weight_decay: float = 0.0,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
    ):
        """
        Perform a single optimization step with a specifiec learning_rate
        and pairs of sentences/labels. Returns loss.
        """

        # Checks
        assert type(sentences) == list, '`sentences` must be a list.'
        assert type(labels) == list, '`labels` must be a list.'

        assert len(sentences) > 0, '`sentences` contains no element.'
        assert len(labels) > 0, '`labels` contains no element.'

        assert len(sentences) == len(labels), \
            '`sentences` and `labels` have different length.'

        # When modifying model weights, invalidate the cacher and point to
        # a temp directory
        self.path_model = Path(
            tempfile.gettempdir()) / f'cacher_{self.create_random_hash()}'
        self.cacher = None

        self.model.zero_grad()
        self.model.train()

        # Tokenize sentences
        inputs = self.tokenizer(sentences, padding=True, return_tensors='pt')

        # Add labels to `inputs`
        if self.exclusive_classes:
            if type(labels[0]) == list:
                # Flatten list of lists for `LabelEncoder`
                labels = [label[0] for label in labels]

            inputs['labels'] = torch.tensor(
                self.label_encoder.transform(labels))
        else:
            inputs['labels'] = torch.tensor(
                self.label_encoder.transform(
                    [tuple(label) for label in labels]))

        if self.device == 'cuda':
            for key in inputs.keys():
                inputs[key] = inputs[key].to('cuda')

        # Forward pass
        outputs = self.model(**inputs)

        loss = outputs[0]

        current_loss = loss.item()

        # Backpropagation of loss
        loss.backward()

        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

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

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=learning_rate,
                          eps=adam_epsilon)

        # Update optimizer
        optimizer.step()

        # Set gradients to zero
        self.model.zero_grad()

        return current_loss
