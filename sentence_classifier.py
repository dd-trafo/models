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
#import re
#import locale
#locale.setlocale(locale.LC_ALL, 'en_US.utf8')
import random
import re
from pathlib import Path

#from utils import *


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


class SentenceClassifier:
    def __init__(
            self,
            df_train: pd.DataFrame,
            df_eval: pd.DataFrame,
            batch_size: int,
            neutral_label: str = 'NEUTRAL',
            path_model:
        str = '/mnt/david_tmp/models/DE/bert-base-german-dbmdz-cased/',
            path_trained_model: str = '/tmp') -> None:

        self.train_sampler = Sampler(df_train,
                                     neutral_label=neutral_label,
                                     balance=True)

        self.eval_sampler = Sampler(df_eval,
                                    neutral_label=neutral_label,
                                    balance=False)

        # eval_sampler = Sampler(df_eval,
        #                        neutral_label=neutral_label,
        #                        balance=False)

        self.labels = self.train_sampler.get_labels()
        self.n_labels = len(self.labels)

        self.label_encoder = MultiLabelBinarizer()
        self.label_encoder.fit([self.labels])

        self.now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

        self.path_model = Path(path_model)
        self.path_trained_model = Path(path_trained_model) / self.now

        # Load model
        self.tokenizer = BertTokenizer.from_pretrained(str(self.path_model))
        self.model = BertForMultiLabelSequenceClassification.from_pretrained(
            str(self.path_model), num_labels=self.n_labels)

        self.weight_decay = 0
        self.warmup_ratio = 0.06
        self.learning_rate = 4e-5
        self.adam_epsilon = 1e-8
        self.device = 'cpu'
        self.max_grad_norm = 1.0
        #save_model_every_epoch = True

        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [{
            'params': [
                p for n, p in self.model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            self.weight_decay,
        }, {
            'params': [
                p for n, p in self.model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.0,
        }]

        self.n_batch = len(df_train) // batch_size

        self.n_batch_eval = len(df_eval) // batch_size

    def train(self, n_epoch: int):

        t_total = batch_size * self.n_batch * n_epoch

        warmup_steps = int(np.ceil(t_total * self.warmup_ratio))

        optimizer = AdamW(self.optimizer_grouped_parameters,
                          lr=self.learning_rate,
                          eps=self.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_total)

        torch.cuda.is_available()

        # Set model to training mode
        self.model.zero_grad()
        self.model.train()

        #device = 'cpu'

        # if device == 'cuda':
        #     self.model.cuda()

        #pbar = trange(n_epoch)

        for i_epoch in range(n_epoch):

            loss_train = 0.0
            i_batch = 0

            pbar = tqdm(self.train_sampler.generator(), total=self.n_batch)
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
                                               self.max_grad_norm)

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
                pbar = tqdm(eval_sampler.generator(batch_size=batch_size),
                            total=self.n_batch_eval)
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

            #save()

    def save(self):
        os.makedirs(self.path_trained_model, exist_ok=True)

        self.model.save_pretrained(self.path_trained_model)
        self.tokenizer.save_pretrained(self.path_trained_model)

        with open(self.path_trained_model / 'label_encoder.pkl',
                  'wb') as handle:
            pickle.dump(label_encoder, handle)

    def predict(self, pred, labels):
        output = dict()
        for i, label in enumerate(labels):
            output[label] = round(float(pred[i]), 3)

        return output

    def logit_to_prob(self, logit):
        return np.exp(logit) / (np.exp(logit) + 1.0)

    def get_hash(s):
        return hashlib.sha256(s.encode('utf-8')).hexdigest()

    def save_feedback(path_feedback, sentence, labels):
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

        df_sentence = pd.DataFrame(data={
            'SENTENCE': [sentence],
            'LABELS': [labels]
        })
        df_sentence.to_csv(path_feedback + get_hash(sentence) + '_' + now +
                           '.txt',
                           index=False)

    def evaluate(self, sentences):
        n_sentence = len(sentences)

        # Tokenize sentences from train_sampler
        inputs = self.tokenizer(sentences, padding=True, return_tensors='pt')

        self.model.eval()
        with torch.no_grad():
            # Evaluate model
            outputs = self.model(**inputs)

            #print(outputs)

            preds = outputs[0].sigmoid().detach().cpu().numpy()

        output = list()
        for i, pred in enumerate(preds):
            output.append((sentences[i], dict(zip(self.labels, pred))))

        return output

    def get_prediction(pred, labels):
        output = dict()
        for i, label in enumerate(labels):
            output[label] = round(float(pred[i]), 3)

        return output

    def cache_prediction(path_cache, hash_, prediction=None):
        """
        Caches prediction to disk or returns content existing cache.
        """

        os.makedirs(path_cache, exist_ok=True)

        if prediction is None:  # load cache
            with open(path_cache + hash_ + '.pkl', 'rb') as handle:
                return pickle.load(handle)
        else:  # store cache
            with open(path_cache + hash_ + '.pkl', 'wb') as handle:
                pickle.dump(prediction, handle)

    def evaluate_model_with_cache(sentences, model, tokenizer, label_encoder,
                                  max_seq_length, device, path_cache):

        n_sentence = len(sentences)
        output = [None] * n_sentence

        pred_indices = list()
        pred_sentences = list()
        pred_hashes = list()

        for i, sentence in enumerate(sentences):
            hash_ = get_hash(sentence)
            cache_file = path_cache + hash_ + '.pkl'

            if os.path.isfile(cache_file):  # load cached prediction
                output[i] = (sentence, cache_prediction(path_cache, hash_))
            else:  # predict
                pred_indices.append(i)
                pred_sentences.append(sentence)
                pred_hashes.append(hash_)

        if len(pred_sentences) > 0:
            predictions = evaluate_model(pred_sentences, model, tokenizer,
                                         label_encoder, max_seq_length, device)

            for i, prediction in enumerate(predictions):
                index = pred_indices[i]
                hash_ = pred_hashes[i]
                sentence = sentences[index]

                # store result
                cache_prediction(path_cache, hash_, prediction[1])

                # put into output
                output[index] = (sentence, prediction[1])

        return output


##################################
##################################
##################################

if __name__ == "__main__":

    path_sentences = '/opt/python/env_huggingface/sentences/DE/'
    #path_feedback = path_sentences + 'feedback/'

    ### load data
    #df = load_sentences(path_sentences, clean=False)

    #df

    ###############
    #labels = get_labels(df, exclude='NEUTRAL')
    #k = len(labels)

    batch_size = 6

    s = SentenceClassifier(df_train=df,
                           df_eval=df,
                           batch_size=batch_size,
                           path_model='/tmp/bert-base-german-dbmdz-cased/')

    s.train(n_epoch=10)

    sentences = ['Ich möchte heiraten.']

    s.evaluate(sentences)

# ### put into model
# labels = train_sampler.get_labels()

# label_encoder = MultiLabelBinarizer()
# label_encoder.fit([labels])

# ######################
# path_model = '/mnt/david_tmp/models/DE/bert-base-german-dbmdz-cased/'
# #path_trained_model = '/opt/python/env_huggingface/trained_model/'
# path_trained_model = '/tmp/trained_model/'

# path_trained_model = path_trained_model + now + '/'

# ### load model
# tokenizer = BertTokenizer.from_pretrained(path_model)
# model = BertForMultiLabelSequenceClassification.from_pretrained(path_model,
#                                                                 num_labels=k)

# #inputs = tokenizer('Hello, my dog is cute', return_tensors='pt')
# #tokenizer('Hello, my dog is cute')

# sampler = Sampler(df, neutral_label='NEUTRAL', balance=True)

# sentences, labels = next(sampler.generator())

# sentences

# inputs = tokenizer(sentences, padding=True, return_tensors='pt')
# inputs['labels'] = label_encoder.transform([tuple(x) for x in labels])

# labels

# labels = sampler.get_labels()
# label_encoder = MultiLabelBinarizer()
# label_encoder.fit([labels])

# x = 'REAL_ESTATE'
# label_encoder.transform([tuple(x.split('|'))])[0]

# [0]

# ### get maximum token length
# max_tokens = get_max_tokens(df, tokenizer)
# print(f'max_tokens: {max_tokens}')

# ### set model parameters
# max_seq_length = 128
# assert max_tokens < max_seq_length, 'Error: max_seq_length is too low. Some sentences would be cut off.'

# weight_decay = 0
# warmup_ratio = 0.06
# learning_rate = 4e-5
# adam_epsilon = 1e-8
# device = 'cpu'
# tr_loss = 0.0
# max_grad_norm = 1.0
# save_model_every_epoch = True

# batch_size = 8
# n_batch = len(df) // batch_size
# n_epoch = 10

# # relevant for scheduler and optimizer
# t_total = batch_size * n_batch * n_epoch

# print('*** Training setup ***')
# print()
# print(f'batch_size: {batch_size:8n}')
# print(f'n_batch:    {n_batch:8n}')
# print(f'n_epoch:    {n_epoch:8n}')
# print(f'--------------------')
# print(f't_total:    {t_total:8n}')
# print()

# no_decay = ['bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [{
#     'params': [
#         p for n, p in model.named_parameters()
#         if not any(nd in n for nd in no_decay)
#     ],
#     'weight_decay':
#     weight_decay,
# }, {
#     'params': [
#         p for n, p in model.named_parameters()
#         if any(nd in n for nd in no_decay)
#     ],
#     'weight_decay':
#     0.0,
# }]

# warmup_steps = int(np.ceil(t_total * warmup_ratio))

# optimizer = AdamW(optimizer_grouped_parameters,
#                   lr=learning_rate,
#                   eps=adam_epsilon)
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps=warmup_steps,
#                                             num_training_steps=t_total)
# generator = batch_index_generator(df=df, batch_size=batch_size, n=batch_size)

# torch.cuda.is_available()

# ### train model
# model.zero_grad()
# model.train()

# if device == 'cuda':
#     model.cuda()

# for i_epoch in trange(n_epoch):
#     loss_epoch = 0
#     for i_batch in range(n_batch):

#         batch_indices = next(generator)

#         inputs = fill_batch(df, batch_indices, labels, batch_size,
#                             max_seq_length, tokenizer, label_encoder, device)

#         outputs = model(**inputs)

#         loss = outputs[0]

#         current_loss = loss.item()
#         loss_epoch = loss_epoch + loss.item()

#         print(f'\nRunning loss: {loss:.8f}', end='')

#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

#         optimizer.step()
#         scheduler.step()  # Update learning rate schedule
#         model.zero_grad()

#     loss_epoch = loss_epoch / n_batch
#     print(f'Average loss: {loss_epoch:.8f}')

#     if save_model_every_epoch:
#         save_model(path_trained_model + f'epoch-{i_epoch}/',
#                    model=model,
#                    tokenizer=tokenizer,
#                    label_encoder=label_encoder)

# if False:
#     ### evaluate model
#     sentence = 'Was für ein regnerisches Wetter!'
#     #sentence = 'Du sollst keiner Statistik glauben, die du nicht selbst gefälscht hast.'
#     #sentence = 'Der Apfel fällt nicht weit vom Stamm.'
#     sentence = 'Das Wetter ist meist bewölkt.'
#     sentence = 'Aktuell liegt keine Wetterwarnung vor.'
#     sentence = 'Es kommt zu Schneefall im Gebirge.'
#     sentence = 'Es besteht Regenrisiko.'

#     feature = sentence_to_feature(sentence, None, max_seq_length, tokenizer,
#                                   label_encoder)

#     inputs = {
#         'input_ids':
#         torch.tensor([feature['input_ids']], dtype=torch.long).to(device),
#         'attention_mask':
#         torch.tensor([feature['attention_mask']], dtype=torch.long).to(device),
#     }

#     ss = SentenceSplitter()

#     ss.split('Das ist ein Text, aber hier geht es weiter.')

#     model.eval()
#     with torch.no_grad():
#         outputs = model(**inputs)

#         logits = outputs[0]

#         logits = logits.sigmoid()

#         preds = logits.detach().cpu().numpy()[0]
#         print(f'preds = {preds}')