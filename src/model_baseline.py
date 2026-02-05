import math
import random
from typing import List, Dict, Any

class FeatureExtractor:
    @staticmethod
    def extract(inp) -> List[float]:
        # Simple feature vector from Pydantic schema
        # 1. recent points avg
        # 2. trend slope
        # 3. days rest
        # 4. opponent def rating
        # 5. implied probability from odds (if available) / line diff
        
        # Access attributes from Pydantic objects or dicts (handling both)
        def get_attr(obj, path, default=0.0):
            try:
                val = obj
                for p in path.split("."):
                    if isinstance(val, dict): val = val.get(p)
                    else: val = getattr(val, p)
                return float(val) if val is not None else default
            except:
                return default

        f1 = get_attr(inp, "player.recent_form.points_avg_5")
        f2 = get_attr(inp, "player.recent_form.trend_points_slope_5")
        f3 = get_attr(inp, "player.workload.days_rest")
        f4 = get_attr(inp, "opponent_team.def_rating", 110.0)
        f5 = get_attr(inp, "target.threshold")
        
        # Interaction: Projection vs Threshold
        # Crude heuristic: (Avg + Slope) - Threshold
        proj = f1 + f2
        diff = proj - f5
        
        return [1.0, diff, f3, f4] # 1.0 is bias term

class BaselineModel:
    def __init__(self, learning_rate=0.01, epochs=100):
        # Weights: [Bias, Diff, DaysRest, OppDef]
        # Initial deterministic weights for "cold start"
        self.weights = [0.0, 0.1, 0.05, -0.01] 
        self.lr = learning_rate
        self.epochs = epochs

    def sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-max(min(z, 10), -10)))

    def predict_proba(self, input_obj) -> float:
        feats = FeatureExtractor.extract(input_obj)
        z = sum(w * x for w, x in zip(self.weights, feats))
        return self.sigmoid(z)
        
    def fit(self, inputs: List[Any], targets: List[int]):
        random.seed(42) # Deterministic behavior
        # Simple SGD Implementation
        data = []
        for x, y in zip(inputs, targets):
            feats = FeatureExtractor.extract(x)
            data.append((feats, float(y)))
            
        for _ in range(self.epochs):
            random.shuffle(data)
            for feats, y in data:
                z = sum(w * xi for w, xi in zip(self.weights, feats))
                pred = self.sigmoid(z)
                err = pred - y
                
                # Update weights: w = w - lr * err * x
                new_weights = []
                for w, xi in zip(self.weights, feats):
                    new_weights.append(w - self.lr * err * xi)
                self.weights = new_weights
