"""
Deterministic local reasoning engine for betting predictions.

Replaces LLM-based reasoning with interpretable, deterministic local scoring.
Outputs structured reasoning including probability, confidence grade, key drivers,
and counter-signals.
"""

from typing import List, Tuple
from src.schemas import PredictionInput, ReasoningOutput
from src.model_baseline import BaselineModel, FeatureExtractor


class LocalReasoningEngine:
    """Deterministic local reasoning engine."""

    def __init__(self):
        """Initialize reasoning engine with baseline model."""
        self.model = BaselineModel()
        # Initialize with small default weights for cold start
        # These will be replaced if model is trained
        self.model.weights = None  # Will initialize on first prediction

    def set_model(self, model: BaselineModel):
        """Set trained model for scoring."""
        self.model = model

    def _extract_key_drivers(self, inp: PredictionInput, features: List[float]) -> List[str]:
        """
        Extract key drivers based on feature values.

        Args:
            inp: PredictionInput object
            features: Extracted feature vector

        Returns:
            List of human-readable key driver strings
        """
        drivers = []

        # Check injury status (feature index 1)
        if features[1] >= 0.9:
            drivers.append("Player fully healthy with no injury concerns")
        elif features[1] < 0.5:
            drivers.append("Injury status raises availability concerns")

        # Check home/away advantage (feature index 3)
        if features[3] == 1.0:
            drivers.append("Home court advantage")
        elif features[3] == 0.0:
            drivers.append("Playing on the road")

        # Check rest (feature index 5)
        if features[5] >= 3.0:
            drivers.append("Well-rested with 3+ days off")
        elif features[5] < 1.0:
            drivers.append("Limited rest could impact performance")

        # Check back-to-back (feature index 6)
        if features[6] == 1.0:
            drivers.append("Second night of back-to-back games")

        # Check matchup advantage (feature index 14)
        matchup_adv = features[14]
        if matchup_adv > 0.5:
            drivers.append("Favorable offensive matchup vs opponent defense")
        elif matchup_adv < -0.5:
            drivers.append("Challenging matchup against strong defense")

        # Check recent form/trend (feature index 18)
        trend = features[18]
        if trend > 0.5:
            drivers.append("Strong positive scoring trend")
        elif trend < -0.5:
            drivers.append("Recent scoring decline")

        # Check pace (feature index 11)
        combined_pace = features[11]
        if combined_pace > 1.02:
            drivers.append("High-pace game environment expected")
        elif combined_pace < 0.98:
            drivers.append("Slower-paced game could limit possessions")

        # Check projection vs threshold (feature index 31)
        differential = features[31]
        if differential > 0.5:
            drivers.append("Recent performance significantly exceeds threshold")
        elif differential < -0.5:
            drivers.append("Threshold set above recent performance levels")

        return drivers[:5]  # Limit to top 5 drivers

    def _extract_counter_signals(self, inp: PredictionInput, features: List[float]) -> List[str]:
        """
        Extract counter-signals (bearish factors).

        Args:
            inp: PredictionInput object
            features: Extracted feature vector

        Returns:
            List of counter-signal strings
        """
        signals = []

        # Injury concerns (feature index 2)
        if features[2] > 0.3:
            signals.append("Injury risk factor present")

        # Travel fatigue (feature index 4)
        if features[4] > 0.5:
            signals.append("High travel fatigue index")

        # Back-to-back (feature index 6)
        if features[6] == 1.0:
            signals.append("Back-to-back game fatigue risk")

        # Tough defender (feature index 15)
        if features[15] > 0.85:
            signals.append("Elite defender in matchup")

        # Poor recent form (feature index 18)
        if features[18] < -0.3:
            signals.append("Negative performance trend")

        # Low usage (feature index 20)
        if features[20] < 0.15:
            signals.append("Low historical usage rate")

        # Unfavorable matchup (feature index 14)
        if features[14] < -0.3:
            signals.append("Defensive matchup disadvantage")

        # Line movement against (feature index 29)
        if abs(features[29]) > 0.5:
            signals.append("Significant betting line movement")

        return signals[:4]  # Limit to top 4 counter-signals

    def _determine_confidence_grade(self, probability: float, drivers: List[str], counters: List[str]) -> str:
        """
        Determine confidence grade based on probability and signal clarity.

        Args:
            probability: Predicted probability
            drivers: List of key drivers
            counters: List of counter-signals

        Returns:
            Confidence grade: A, B, C, D, or F
        """
        # Start with probability-based grading
        if probability >= 0.65 or probability <= 0.35:
            # Strong directional signal
            base_grade = "A"
        elif probability >= 0.58 or probability <= 0.42:
            base_grade = "B"
        elif probability >= 0.53 or probability <= 0.47:
            base_grade = "C"
        else:
            base_grade = "D"

        # Downgrade if conflicting signals
        signal_ratio = len(drivers) / max(len(counters), 1)
        if signal_ratio < 1.2 and base_grade in ["A", "B"]:
            # Many counter-signals relative to drivers
            grade_map = {"A": "B", "B": "C", "C": "D", "D": "F"}
            base_grade = grade_map.get(base_grade, "F")

        return base_grade

    def generate_reasoning(self, inp: PredictionInput) -> ReasoningOutput:
        """
        Generate deterministic reasoning output.

        Args:
            inp: PredictionInput object

        Returns:
            ReasoningOutput with probability, confidence, drivers, and counter-signals
        """
        # Extract features
        features = FeatureExtractor.extract(inp)

        # Get probability from model
        probability = self.model.predict_proba(inp)

        # Extract key drivers and counter-signals
        key_drivers = self._extract_key_drivers(inp, features)
        counter_signals = self._extract_counter_signals(inp, features)

        # Determine confidence grade
        confidence_grade = self._determine_confidence_grade(probability, key_drivers, counter_signals)

        # Generate thesis
        event_type = inp.target.event_type
        threshold = inp.target.threshold
        player_name = inp.player.player_id  # In real scenario, would have player name

        direction = "over" if probability > 0.5 else "under"
        confidence_desc = {
            "A": "high",
            "B": "moderate-high",
            "C": "moderate",
            "D": "low-moderate",
            "F": "low"
        }.get(confidence_grade, "uncertain")

        thesis = (
            f"Deterministic model predicts {direction} {threshold} for {event_type} "
            f"with {confidence_desc} confidence (p={probability:.3f}). "
            f"Analysis based on {len(features)} comprehensive features."
        )

        # Uncertainty notes
        uncertainty_notes = []
        if 0.45 <= probability <= 0.55:
            uncertainty_notes.append("Probability near 50% indicates high uncertainty")
        if len(counter_signals) >= len(key_drivers):
            uncertainty_notes.append("Mixed signals present - both bullish and bearish factors")
        if confidence_grade in ["D", "F"]:
            uncertainty_notes.append("Low confidence grade suggests avoiding this bet")

        return ReasoningOutput(
            thesis=thesis,
            probability=round(probability, 4),
            confidence_grade=confidence_grade,
            key_drivers=key_drivers if key_drivers else ["Baseline model prediction"],
            counter_signals=counter_signals if counter_signals else [],
            uncertainty_notes=uncertainty_notes
        )


# Global engine instance
_reasoning_engine = LocalReasoningEngine()


def generate_reasoning(inp: PredictionInput) -> ReasoningOutput:
    """
    Generate reasoning for prediction input.

    Args:
        inp: PredictionInput object

    Returns:
        ReasoningOutput with deterministic local scoring
    """
    return _reasoning_engine.generate_reasoning(inp)


def set_reasoning_model(model: BaselineModel):
    """
    Set trained model for reasoning engine.

    Args:
        model: Trained BaselineModel
    """
    _reasoning_engine.set_model(model)