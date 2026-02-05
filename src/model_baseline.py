import math
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class FeatureExtractor:
    """
    Feature extractor using comprehensive feature set.

    Supports both Pydantic objects and CSV-based features via delegation
    to src.features.FeatureExtractor.
    """

    @staticmethod
    def extract(inp) -> List[float]:
        """
        Extract comprehensive feature vector.

        Delegates to src.features.FeatureExtractor for rich feature extraction
        when available, falls back to simple extraction otherwise.
        """
        try:
            from src.features import FeatureExtractor as CSVFeatureExtractor
            extractor = CSVFeatureExtractor()
            return extractor.extract_from_pydantic(inp)
        except:
            # Fallback to simple feature extraction if CSV data not available
            return FeatureExtractor._extract_simple(inp)

    @staticmethod
    def _extract_simple(inp) -> List[float]:
        """Simplified feature extraction when CSV data is unavailable."""
        def get_attr(obj, path, default=0.0):
            try:
                val = obj
                for p in path.split("."):
                    if isinstance(val, dict):
                        val = val.get(p)
                    else:
                        val = getattr(val, p, None)
                return float(val) if val is not None else default
            except:
                return default

        # Extract basic features
        f1 = get_attr(inp, "player.recent_form.points_avg_5", 20.0)
        f2 = get_attr(inp, "player.recent_form.trend_points_slope_5", 0.0)
        f3 = get_attr(inp, "player.workload.days_rest", 2.0)
        f4 = get_attr(inp, "opponent_team.def_rating", 110.0)
        f5 = get_attr(inp, "target.threshold", 0.0)

        proj = f1 + f2
        diff = proj - f5

        # Return minimal feature set (4 features)
        return [1.0, diff, f3, f4]


class BaselineModel:
    """
    Baseline logistic regression model with chronological train/val split.

    Features:
    - Supports comprehensive 32-feature vectors from src.features
    - Chronological splitting to prevent temporal leakage
    - Simple SGD training with deterministic behavior
    """

    def __init__(self, learning_rate=0.01, epochs=50, val_split_date: Optional[str] = None):
        """
        Initialize baseline model.

        Args:
            learning_rate: Learning rate for SGD
            epochs: Number of training epochs
            val_split_date: ISO date string for train/val split (e.g., "2024-01-25")
        """
        # Weights initialized to zeros - will be sized on first fit
        self.weights = None
        self.lr = learning_rate
        self.epochs = epochs
        self.val_split_date = val_split_date
        self.feature_dim = None

    def sigmoid(self, z: float) -> float:
        """Numerically stable sigmoid function."""
        return 1.0 / (1.0 + math.exp(-max(min(z, 10), -10)))

    def predict_proba(self, input_obj) -> float:
        """
        Predict probability for input.

        Args:
            input_obj: Input object (Pydantic or dict)

        Returns:
            Probability between 0 and 1
        """
        feats = FeatureExtractor.extract(input_obj)

        # Initialize weights if not yet set
        if self.weights is None:
            self.feature_dim = len(feats)
            self.weights = [0.0] * self.feature_dim

        z = sum(w * x for w, x in zip(self.weights, feats))
        return self.sigmoid(z)

    def fit(self, inputs: List[Any], targets: List[int], dates: Optional[List[str]] = None):
        """
        Fit model with optional chronological split.

        Args:
            inputs: List of input objects
            targets: List of binary targets (0 or 1)
            dates: Optional list of ISO date strings for chronological splitting
        """
        random.seed(42)  # Deterministic behavior

        # Extract features
        data = []
        for i, (x, y) in enumerate(zip(inputs, targets)):
            feats = FeatureExtractor.extract(x)
            date = dates[i] if dates and i < len(dates) else None
            data.append((feats, float(y), date))

        # Initialize weights based on feature dimension
        if not data:
            return

        self.feature_dim = len(data[0][0])
        if self.weights is None:
            self.weights = [0.0] * self.feature_dim

        # Chronological split if date provided
        train_data, val_data = self._chronological_split(data)

        # Train on training set
        for epoch in range(self.epochs):
            random.shuffle(train_data)
            for feats, y, _ in train_data:
                z = sum(w * xi for w, xi in zip(self.weights, feats))
                pred = self.sigmoid(z)
                err = pred - y

                # Update weights: w = w - lr * err * x
                new_weights = []
                for w, xi in zip(self.weights, feats):
                    new_weights.append(w - self.lr * err * xi)
                self.weights = new_weights

        # Optionally evaluate on validation set
        if val_data and len(val_data) > 0:
            self._evaluate_validation(val_data)

    def _chronological_split(
        self,
        data: List[Tuple[List[float], float, Optional[str]]]
    ) -> Tuple[List, List]:
        """
        Split data chronologically.

        Args:
            data: List of (features, target, date) tuples

        Returns:
            (train_data, val_data) tuple
        """
        if not self.val_split_date:
            # No split, use all data for training
            return data, []

        train_data = []
        val_data = []

        for feats, y, date in data:
            if date is None:
                # If no date, add to training
                train_data.append((feats, y, date))
            elif date < self.val_split_date:
                train_data.append((feats, y, date))
            else:
                val_data.append((feats, y, date))

        return train_data, val_data

    def _evaluate_validation(self, val_data: List[Tuple[List[float], float, Optional[str]]]):
        """Evaluate model on validation set (for monitoring only)."""
        if not val_data:
            return

        correct = 0
        total = 0

        for feats, y, _ in val_data:
            z = sum(w * xi for w, xi in zip(self.weights, feats))
            pred_prob = self.sigmoid(z)
            pred_class = 1 if pred_prob >= 0.5 else 0

            if pred_class == int(y):
                correct += 1
            total += 1

        # Validation accuracy (not used for training, just monitoring)
        if total > 0:
            accuracy = correct / total
            # Note: In production, this would be logged, not printed
