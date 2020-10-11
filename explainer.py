import itertools
import re
from typing import Dict, List, Any, Union, Type, Tuple, Callable

import numpy as np
from scipy.stats import entropy
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertTokenizerFast


class Explainer:
    def _cosine_similarity(self, n_tokens: int, n_removal: int) -> np.float:
        """
        Return cosine similarity between a binary vector with all ones
        of length `n_tokens and vectors of the same length with
        `n_removal` elements set to zero.
        """

        remaining = -np.array(n_removal) + n_tokens
        return remaining / (np.sqrt(n_tokens + 1e-6) *
                            np.sqrt(remaining + 1e-6))

    def _mean_kl_divergence(self,
                            y_pred,
                            y,
                            sample_weight=None,
                            eps=1e-9) -> np.float:
        """
        Calculate mean KL divergence between y_pred and y
        """

        kl_elementwise = entropy(y.T, y_pred.T + eps)
        return np.average(kl_elementwise, weights=sample_weight)

    def _remove_prefix(self, txt: str, prefix: str = '##') -> str:
        """
        Remove wordpiece prefix from the beginning of a string
        """

        # TODO: wordpiece_prefix is hardcoded for now, derive from tokenizer
        return re.sub(f'^{prefix}', '', txt.strip())

    def _sample(self,
                tokens: np.array,
                min_removal: int = 1,
                max_removal: int = 3):
        """
        Return all combinatorial combinations of token removals
        """

        n_tokens = len(tokens)

        # Iterate over all combinatorial combinations of an `n_tokens`
        # number of binary variables, i.e.
        # [
        #   [0, 0, 0]
        #   [0, 0, 1]
        #   [0, 1, 0]
        #   ...
        #   [1, 1, 0]
        #   [1, 1, 1]
        # ]

        X_list = list()
        tokens_list = list()
        sims_list = list()

        for idx_to_remove in list(itertools.product([0, 1], repeat=n_tokens)):
            # Count how many tokens would this remove
            n_removal = sum(idx_to_remove)
            if (n_removal >= min_removal) and (n_removal <= max_removal):
                # `mask` is a binary vector where 1s represent tokens to be removed
                mask = np.array(idx_to_remove)

                # `indicator` is the inverse of `mask`
                indicator = 1 - mask
                X_list.append(indicator)

                # Remove tokens defined by `mask`
                tokens_ = np.delete(tokens, mask.astype(bool))
                tokens_list.append(tokens_)

                # Calculate similarity between `tokens` and `tokens_`
                sim = self._cosine_similarity(n_tokens, n_tokens - n_removal)
                sims_list.append(sim)

        # Convert lists into numpy arrays
        X = np.array(X_list)
        sims = np.array(sims_list)

        return tokens_list, X, sims

    def expand(self,
               X: np.array,
               y: np.array,
               sims: np.array,
               n_expand: int = 10) -> Tuple[np.array, np.array, np.array]:
        """
        Since SDGClassifier requires binary targets instead of probabilities,
        we will replicate rows in X and sim by `n_repeat` times and assign them
        labels corresponding to y[i], i.e. y[i] = 0.6 will result in 60% 1s and
        40%s 0s.
        """

        assert X.shape[0] == y.shape[0] == sims.shape[
            0], f'Dimension 0 of X ({X.shape[0]}), y ({y.shape[0]}) and sims ({sims.shape[0]}) do not match.'

        X_list = list()
        y_list = list()
        sims_list = list()

        for i in range(len(X)):
            X_ = np.repeat(X[[i]], n_expand, axis=0)
            y_ = np.zeros(shape=n_expand, dtype=int)
            y_[0:int(y[i] * n_expand)] = 1
            sims_ = np.repeat(sims[i], n_expand)

            X_list.append(X_)
            y_list.append(y_)
            sims_list.append(sims_)

        X_expanded = np.concatenate(X_list)
        y_expanded = np.concatenate(y_list)
        sims_expanded = np.concatenate(sims_list)

        return X_expanded, y_expanded, sims_expanded

    def explain(self,
                txt: str,
                tokenizer: Type[BertTokenizer],
                predict: Callable,
                label: str,
                min_removal: int = 1,
                max_removal: int = 3,
                n_max: Union[None, int] = 96,
                test_size: Union[None, float] = 0.3,
                use_cache: bool = True,
                n_expand: int = 10):
        """
        Explain the output of a black-box classifier for a given string `txt`
        by imitating its local behavior with a white-box classifier. Samples
        are generated from all combinations that the supplied `tokenizer`
        distinguishes.
        """

        # Check parameters
        assert len(txt.strip()) > 0, '`txt` is empty.'
        assert min_removal >= 1, '`min_removal` must be greater or equal 1.'
        assert min_removal <= max_removal, '`min_removal` must be less or equal `max_removal`.'
        assert n_expand >= 1, '`n_repeat` must be greater or equal 1.'
        assert label is not None, '`label` must be supplied.'
        assert (n_max is None) or (
            n_max >
            0), f'`n_max` ({n_max}) must either be None of greater zero.'

        # Tokenize `txt`
        tokens = tokenizer.encode(txt, add_special_tokens=False)
        tokens = np.array(tokens)

        # Get each word that the tokenizer distinguishes
        words = [
            self._remove_prefix(tokenizer.decode([token])) for token in tokens
        ]

        # Generate variations of `txt` by removing tokens
        tokens_list, X, sims = self._sample(tokens,
                                            min_removal=min_removal,
                                            max_removal=max_removal)

        # We received this many variations
        n_samples = len(tokens_list)

        if n_max is None:
            print(f'INFO: Evaluating {n_samples} samples.')

        if (n_max is not None) and (n_max < n_samples):
            # Evaluate `n_max` out of `n_samples` samples. First, shuffle
            # generated samples to get a more diverse selection.
            tokens_list, X, sims = shuffle(tokens_list,
                                           X,
                                           sims,
                                           random_state=42)

            # Translate the first `n_max` samples into strings
            txts_ = [
                self._remove_prefix(tokenizer.decode(tokens_))
                for tokens_ in tokens_list[0:n_max]
            ]

            # Evaluate `N_MAX` strings by supplied predictor
            preds = predict(txts_, use_cache=use_cache)

            # Extract the given label from the prediction
            y = np.array([pred['LABELS'][label] for pred in preds])

            # Slice X and sims accordingly
            X = X[0:n_max]
            sims = sims[0:n_max]
        else:
            # Evaluate all generated samples. First, translate all token
            # variations into strings
            txts = [
                self._remove_prefix(tokenizer.decode(tokens_))
                for tokens_ in tokens_list
            ]
            preds = predict(txts, use_cache=use_cache)

            y = np.array([pred['LABELS'][label] for pred in preds])

        # Check if X, y and sims have correct shape
        assert X.shape[0] == y.shape[0] == sims.shape[
            0], f'Dimension 0 of X ({X.shape[0]}), y ({y.shape[0]}) and sims ({sims.shape[0]}) do not match.'

        if test_size is None:
            X_train = X
            y_train = y
            sims_train = sims
        else:
            # Split X, y and sims into train and test sets
            X_train, X_test, y_train, y_test, sims_train, sims_test = train_test_split(
                X, y, sims, test_size=test_size, random_state=42)

        # SDGClassifier requires binary y_train, thus, transform X_train,
        # y_train and sims_train
        X_train_expanded, y_train_expanded, sims_train_expanded = self.expand(
            X_train, y_train, sims_train, n_expand=n_expand)

        # We need to have predictions for each of 2 classes
        assert len(
            np.unique(y_train)
        ) > 1, 'Black-box model only predicted a single class for all derived samples, consider increasing `n_max`.'

        # Create white-box model (logistic regression)
        clf = SGDClassifier(loss='log',
                            penalty='elasticnet',
                            alpha=1e-3,
                            random_state=42)

        # Train model
        clf.fit(X_train_expanded,
                y_train_expanded,
                sample_weight=sims_train_expanded)

        weights = dict(zip(words, clf.coef_[0]))

        if test_size is None:
            kld = None
            score = None
        else:
            # Predict test set
            y_test_pred = clf.predict_proba(X_test)

            # Evaluate KL divergence
            y_test_ = np.c_[1.0 - y_test, y_test]
            kld = self._mean_kl_divergence(y_test_pred,
                                           y_test_,
                                           sample_weight=sims_test)

            # Compute score of test set
            score = clf.score(X_test,
                              y_test_.argmax(axis=1),
                              sample_weight=sims_test)

        return {
            'weights': weights,
            'label': label,
            'kld': kld,
            'score': score,
            'n_samples': n_samples,
            'n_samples_evaluated': len(X)
        }
