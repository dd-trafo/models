import numpy as np
import pandas as pd
import random
from typing import Generator
import string


class Sampler:
    def __init__(self,
                 df: pd.DataFrame,
                 neutral_label: str = 'NEUTRAL',
                 balance: bool = True,
                 use_weights: bool = False) -> None:

        self.balance = balance
        self.use_weights = use_weights

        # Perform checks
        if 'SENTENCE' not in df.columns:
            raise Exception('df must contain column "SENTENCE"')

        if 'LABELS' not in df.columns:
            raise Exception('df must contain column "LABELS"')

        if self.use_weights:
            if 'P' not in df.columns:
                raise Exception(
                    'df must contain column "P" when option "use_weights" == True'
                )

        if (self.balance == False) and self.use_weights:
            raise Exception('Option "use_weights" requires "balance" == True')

        self._n = len(df)
        if self._n < 1:
            raise Exception('df is empty')

        # Sampling requires unique indices, hence reset them
        df = df.reset_index(drop=True)

        # Split df's columns into individual parts
        self.df_sentence = df['SENTENCE']
        self.df_labels = df['LABELS'].str.split('|')
        if self.use_weights:
            self.df_p = df['P'].astype(float)

        # neutral_label will be excluded from self._labels
        self.neutral_label = neutral_label

        # Set the following properties for performance reasons
        self._labels = sorted(
            set([item for sublist in self.df_labels for item in sublist]))

        self._n_labels = len(self._labels)

        self._labels_encoded = [
            string.ascii_uppercase[i] for i in range(self._n_labels)
        ]

        self._labels_wo_neutral = [
            label for label in self._labels if label != self.neutral_label
        ]

        # Since each label is represented as a character in
        # strings.ascii_uppercase, the number is limited
        if self._n_labels > len(string.ascii_uppercase):
            raise Exception(
                f'df contains too many labels ({self.n_labels_w_neutral} > {len(string.ascii_uppercase)})'
            )

        self.df_labels_encoded = self.df_labels.apply(
            lambda labels: self._encode_labels(labels))

    def _encode_labels(self, labels: list) -> str:
        """
        Each sentence has a list of labels associated. Represent each label with a character and concatenate them into a string for fast checks.
        """

        labels_encoded = [
            string.ascii_uppercase[self._labels.index(label)]
            for label in labels
        ]

        return ''.join(labels_encoded)

    def get_labels(self, include_neutral_label=False) -> list:
        if include_neutral_label:
            return self._labels
        else:
            return self._labels_wo_neutral

    def print_stats(self) -> None:
        """
        Show label statictics based on supplied df.
        """

        print()
        print('*** Label Statistic ***')
        print()
        for label in self._labels:
            n_ = len(self.df_labels[self.df_labels.apply(
                lambda labels: any(item in label for item in labels))])

            print(
                f'{label:<15s}: {n_:8n} / {self.n:n} ({(n_ / self.n) * 100:6.2f}%)'
            )

        print()

    def modify_weights(self, func):
        if self.use_weights == False:
            raise Exception(
                'Can only modify weights if sampler is initialized with "use_weights" == True'
            )

        self.df_p = self.df_p.apply(func)

    def _sample_balanced_indices(self, batch_size: int) -> list:
        """
        Return a set of indices where each label has an equal number of occurences.
        """

        # Pick 'batch_size' sentences randomly with replacement by...
        # ... sampling according to probabilities given by self.df_p
        if self.use_weights:
            indices = list(
                np.random.choice(self.df_sentence.index,
                                 size=batch_size,
                                 replace=True,
                                 p=self.df_p / self.df_p.sum()))
        # ... sampling uniformly, every sentence is picked with same probability
        else:
            indices = list(
                np.random.randint(low=0, high=self._n, size=batch_size))

        # Count label occurrences within batch_indices
        value_counts = self.df_labels_encoded.loc[indices].value_counts()

        # Assign counts for multi-label elements to their individual labels each
        # e.g. a sentence with label [A, B] is counted as 1x sentence for A and
        # 1x sentence for B
        multi_labels = [
            label for label in value_counts.index if len(label) > 1
        ]
        for multi_label in multi_labels:
            for label in list(multi_label):
                if label not in value_counts.index:
                    value_counts[label] = 0
                else:
                    value_counts[label] = value_counts[label] + value_counts[
                        multi_label]

        # This random selection may not cover all labels, hence add them here
        value_counts = value_counts.reindex(
            self._labels_encoded).fillna(0).astype(int)

        # This is the upper limit of occurrence for any label
        max_occurrence = value_counts.max()

        # Keep a dictionary for easy tracking of counts
        counts = value_counts.to_dict()

        del value_counts

        # Determine labels that we have enough of already
        labels_to_exclude = [
            label for label, value in counts.items() if value == max_occurrence
        ]

        # Test if the above selection requires balancing/oversampling
        if len(labels_to_exclude) < self._n_labels:
            # Estimate maximum number of steps needed (assuming
            # sentence-to-label is 1:1). We may undercut this.
            steps = 0
            for label, value in counts.items():
                steps = steps + (max_occurrence - value)

            df_labels_encoded_ = self.df_labels_encoded.copy()
            for _ in range(steps):

                # Determine the remaining pool of sentences we may pick from
                df_labels_encoded_ = df_labels_encoded_[
                    ~df_labels_encoded_.str.contains('|'.join(labels_to_exclude
                                                              ))]

                n_ = len(df_labels_encoded_)

                # No element left to pick from, stop
                if n_ == 0:
                    break

                # Randomly select one sentence based on the supplied weights
                if self.use_weights:
                    # Get weights for the remaining pool of sentences
                    p_ = self.df_p.loc[df_labels_encoded_.index]

                    index_random = np.random.choice(n_, p=p_ / p_.sum())

                # Uniform randomly select one sentence
                else:
                    index_random = np.random.randint(low=0, high=n_)

                # Add this selection to 'indices'
                indices.append(df_labels_encoded_.index[index_random])

                # Increase corresponding count(s) of labels
                # Note: a sentence may have multiple labels, hence we iterate over of them
                for label in list(df_labels_encoded_.iloc[index_random]):
                    counts[label] = counts[label] + 1

                    # Check if we already have enough of 'label'
                    if counts[label] == max_occurrence:
                        labels_to_exclude.append(label)

                # All labels have been selected equally, stop
                if len(labels_to_exclude) == self._n_labels:
                    break

            # Shuffle batch_indices because we added elements deterministically
            random.shuffle(indices)

        return indices

    def generator(self, batch_size: int = 8):  # -> Generator[list, list]:
        """
        Return a fixed-sized (batch_size) list based on _sample_batch_indices().
        """

        assert batch_size > 0, f'Error: batch_size must a positive number (given: {batch_size})'

        n_batch = np.ceil(self._n / batch_size).astype(int)

        # Make sure n_batch is at least one, e.g. when batch_size is greater
        # than self._n
        n_batch = max(n_batch, 1)

        # Generate a series of batches where every class appears in equal numbers,
        # e.g. for feeding training data to a neural network
        if self.balance:
            indices = self._sample_balanced_indices(batch_size)
            n_indices = len(indices)

            # Set the initial counter
            i = 0

            # Generate 'n_batch' batches of size 'batch_size', then stop
            for _ in range(n_batch):
                # Fill batch_indices
                batch_indices = indices[i:(i + batch_size)]

                if (i + batch_size) == (n_indices - 1):
                    # We picked the last element, i.e. indices is exhausted,
                    # let's resample
                    i = 0
                    indices = self._sample_balanced_indices(batch_size)
                    n_indices = len(indices)
                elif len(batch_indices) == batch_size:
                    # Standard case: we picked elements, but 'indices' is not exhausted
                    i = i + batch_size
                else:
                    # We filled the current batch only partly,
                    # let's resample and add remaining elements until batch is full
                    while len(batch_indices) < batch_size:
                        delta = batch_size - len(batch_indices)

                        # Resample
                        indices = self._sample_balanced_indices(batch_size)
                        n_indices = len(indices)

                        batch_indices.extend(indices[0:delta])
                        i = delta

                random.shuffle(batch_indices)

                batch_sentences = self.df_sentence.loc[batch_indices].to_list()
                batch_labels = self.df_labels.loc[batch_indices].to_list()

                yield batch_sentences, batch_labels
        # Serve all data in batches without any balancing, e.g. for evaluation
        else:
            indices = list(range(self._n))
            random.shuffle(indices)

            # Set the initial counter
            i = 0

            # Generate 'n_batch' batches of size 'batch_size', then stop
            for _ in range(n_batch):
                # Fill 'batch_indices'
                batch_indices = indices[i:(i + batch_size)]

                # Increment counter 'i' by the number of elements just picked
                i = i + batch_size

                # Shuffle (again) each batch to be sure there are no patterns
                random.shuffle(batch_indices)

                batch_sentences = self.df_sentence.loc[batch_indices].to_list()
                batch_labels = self.df_labels.loc[batch_indices].to_list()

                yield batch_sentences, batch_labels


if __name__ == "__main__":

    df['P'] = 1  #np.random.randint(low=0, high=3, size=len(df)).astype(float)
    df.loc[0, 'P'] = 1000
    df.loc[1, 'P'] = 10
    #df['P'] = df['P'] / df['P'].sum()

    s = Sampler(df, balance=True, use_weights=True)

    a, b = next(s.generator())

    #s.generate()

    # batch_size = 8

    # for a, b in s.generator(batch_size=batch_size):
    #     print(a)
    #     print('-------')

    #sampler.reset()

    # x = list()
    # for i in trange(1000):
    #     x.extend(sampler.generate()[1])
    #     #g.generate()[1]

    # print(pd.Series(x).value_counts())

# pd.Series(x).value_counts()

# list(set(x.split('|')))

# a = pd.concat(x, ignore_index=True)

# a.value_counts()