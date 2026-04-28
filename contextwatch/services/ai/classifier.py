"""Custom k-Nearest Neighbors classifier implementation.

This module implements a simple k-NN classifier from scratch using only
Python's standard library and the custom vector math functions we've
developed.
"""

from __future__ import annotations

from typing import List, Tuple, Any, Optional
from contextwatch.utils.vector_math import cosine_distance


class KNearestNeighbors:
    def __init__(self, k: int = 3, weighted: bool = True):
        """Initialize k-NN classifier.

        Parameters:
        -----------
        k: int
            Number of neighbors to consider
        weighted: bool
            Whether to use distance-weighted voting (closer neighbors have more influence)
        """
        self.k = k
        self.weighted = weighted
        self.X_train: List[List[float]] = []
        self.y_train: List[Any] = []
        self.classes_: List[Any] = []

    def fit(self, X: List[List[float]], y: List[Any]) -> 'KNearestNeighbors':
        """Fit the model using training data.

        Parameters:
        -----------
        X: List of feature vectors
        y: List of corresponding labels

        Returns:
        --------
        self: Returns the instance for method chaining
        """
        if len(X) != len(y):
            raise ValueError("Number of samples in X and y must match")

        self.X_train = X
        self.y_train = y
        self.classes_ = sorted(list(set(y)))
        return self

    def predict(self, X: List[List[float]]) -> List[Any]:
        """Predict class labels for samples in X.

        Parameters:
        -----------
        X: List of feature vectors to predict

        Returns:
        --------
        List of predicted class labels
        """
        if not self.X_train:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        predictions = []
        for sample in X:
            predictions.append(self._predict_single(sample))
        return predictions

    def _predict_single(self, sample: List[float]) -> Any:
        """Predict class for a single sample."""
        # Compute distances to all training samples
        distances = []
        for idx, train_sample in enumerate(self.X_train):
            dist = cosine_distance(sample, train_sample)
            distances.append((dist, self.y_train[idx]))

        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[0])

        # Take k nearest neighbors
        k_nearest = distances[:self.k]

        if self.weighted:
            # Weighted voting: weight = 1 / (distance + epsilon) to avoid division by zero
            epsilon = 1e-8
            class_votes = {}
            total_weight = 0.0

            for distance, label in k_nearest:
                weight = 1.0 / (distance + epsilon)
                class_votes[label] = class_votes.get(label, 0.0) + weight
                total_weight += weight

            # Return class with highest weighted vote
            return max(class_votes.items(), key=lambda x: x[1])[0]
        else:
            # Simple majority vote
            from collections import Counter
            votes = [label for _, label in k_nearest]
            return Counter(votes).most_common(1)[0][0]

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """Predict class probabilities for samples in X.

        Parameters:
        -----------
        X: List of feature vectors

        Returns:
        --------
        List of probability arrays (one per sample)
        """
        if not self.X_train:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        probas = []
        for sample in X:
            proba = self._predict_proba_single(sample)
            probas.append(proba)
        return probas

    def _predict_proba_single(self, sample: List[float]) -> List[float]:
        """Predict class probabilities for a single sample."""
        # Compute distances to all training samples
        distances = []
        for idx, train_sample in enumerate(self.X_train):
            dist = cosine_distance(sample, train_sample)
            distances.append((dist, self.y_train[idx]))

        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[0])

        # Take k nearest neighbors
        k_nearest = distances[:self.k]

        if self.weighted:
            # Weighted voting for probabilities
            epsilon = 1e-8
            class_votes = {cls: 0.0 for cls in self.classes_}
            total_weight = 0.0

            for distance, label in k_nearest:
                weight = 1.0 / (distance + epsilon)
                class_votes[label] = class_votes.get(label, 0.0) + weight
                total_weight += weight

            # Normalize to probabilities
            if total_weight > 0:
                return [class_votes.get(cls, 0.0) / total_weight for cls in self.classes_]
            else:
                # Fallback: uniform distribution
                return [1.0 / len(self.classes_)] * len(self.classes_)
        else:
            # Simple counting for probabilities
            from collections import Counter
            votes = [label for _, label in k_nearest]
            vote_counts = Counter(votes)
            total = len(k_nearest)
            return [vote_counts.get(cls, 0) / total for cls in self.classes_]

"""End of KNearestNeighbors classifier."""
