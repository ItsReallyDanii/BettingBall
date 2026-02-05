import math
from typing import List

class PlattScaler:
    def __init__(self, lr=0.01, epochs=1000):
        self.a = 1.0
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    def fit(self, probs: List[float], outcomes: List[int]):
        # Optimize Log Loss w.r.t parameters a and b
        # P(y|p) = sigmoid(a * logit(p) + b)
        # where logit(p) = ln(p / (1-p))
        
        epsilon = 1e-15
        logits = []
        for p in probs:
            p_safe = max(epsilon, min(1 - epsilon, p))
            logits.append(math.log(p_safe / (1 - p_safe)))
            
        # SGD
        data = list(zip(logits, outcomes))
        n = len(data)
        if n == 0: return

        for _ in range(self.epochs):
            total_grad_a = 0.0
            total_grad_b = 0.0
            
            for logit_val, y in data:
                z = self.a * logit_val + self.b
                sigma = 1.0 / (1.0 + math.exp(-max(min(z, 10), -10)))
                
                # Gradient of Log Loss
                err = sigma - y
                total_grad_a += err * logit_val
                total_grad_b += err
                
            self.a -= self.lr * (total_grad_a / n)
            self.b -= self.lr * (total_grad_b / n)

    def predict(self, prob: float) -> float:
        epsilon = 1e-15
        p_safe = max(epsilon, min(1 - epsilon, prob))
        logit = math.log(p_safe / (1 - p_safe))
        z = self.a * logit + self.b
        return 1.0 / (1.0 + math.exp(-max(min(z, 10), -10)))

class IsotonicScaler:
    """Simple binned isotonic regression (monotonic bin averages)."""
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_values = [0.5] * n_bins

    def fit(self, probs: List[float], outcomes: List[int]):
        if not probs: return
        
        bins = [[] for _ in range(self.n_bins)]
        for p, y in zip(probs, outcomes):
            idx = min(int(p * self.n_bins), self.n_bins - 1)
            bins[idx].append(y)
            
        # Compute bin averages
        averages = []
        for b in bins:
            if b:
                averages.append(sum(b) / len(b))
            else:
                averages.append(None)
                
        # Interpolate missing bins
        last_val = 0.0
        for i in range(self.n_bins):
            if averages[i] is None:
                # Find next available
                next_val = None
                for j in range(i + 1, self.n_bins):
                    if averages[j] is not None:
                        next_val = averages[j]
                        break
                if next_val is not None:
                    averages[i] = (last_val + next_val) / 2
                else:
                    averages[i] = last_val
            last_val = averages[i]
            
        # Force monotonicity
        for i in range(1, self.n_bins):
            if averages[i] < averages[i-1]:
                averages[i] = averages[i-1]
                
        self.bin_values = averages

    def predict(self, prob: float) -> float:
        idx = min(int(prob * self.n_bins), self.n_bins - 1)
        return self.bin_values[idx]
