from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional
from enum import Enum
from datetime import datetime

class InjuryStatus(str, Enum):
    HEALTHY = "healthy"
    QUESTIONABLE = "questionable"
    DOUBTFUL = "doubtful"
    OUT = "out"

class Availability(BaseModel):
    injury_status: InjuryStatus
    probable_minutes_cap: Optional[float] = None

class RecentForm(BaseModel):
    points_avg_5: float
    trend_points_slope_5: float
    trend_ts_slope_5: float

class Workload(BaseModel):
    days_rest: float
    back_to_back: bool
    travel_distance_km_recent: float

class PlayerEntity(BaseModel):
    player_id: str
    team_id: str
    position: str
    recent_form: RecentForm
    workload: Workload
    availability: Availability

class TeamEntity(BaseModel):
    team_id: str
    pace: float
    off_rating: float
    def_rating: float
    rotation_stability: float
    net_rating_last_10: Optional[float] = None
    turnover_pct: Optional[float] = None
    rebound_pct: Optional[float] = None
    three_rate: Optional[float] = None
    rim_freq: Optional[float] = None
    bench_minutes_share: Optional[float] = None

class SpreadContext(BaseModel):
    market_open_line: float
    market_current_line: float
    line_move_abs: float

class GameContextEntity(BaseModel):
    game_id: str
    date_utc: datetime
    home_team_id: str
    away_team_id: str
    is_home_for_subject_team: bool
    days_rest_team: float
    days_rest_opponent: float
    head_to_head_last_3: float
    defender_matchup_index: float
    ref_crew_foul_tendency: float
    travel_fatigue_index: float
    spread_context: SpreadContext

class TargetEntity(BaseModel):
    event_type: Literal["player_points_over", "player_assists_over", "player_rebounds_over", "team_total_over", "spread_cover"]
    threshold: float
    horizon: Literal["pregame"] = "pregame"

    @model_validator(mode='after')
    def validate_threshold(self) -> 'TargetEntity':
        if self.event_type.startswith("player_") or self.event_type == "team_total_over":
            if not (0 < self.threshold <= 100):
                raise ValueError("threshold must be in range (0, 100]")
        return self

class PredictionInput(BaseModel):
    player: PlayerEntity
    team: TeamEntity
    opponent_team: TeamEntity
    game_context: GameContextEntity
    target: TargetEntity
    computed_at: datetime

    @model_validator(mode='after')
    def validate_temporal_isolation(self) -> 'PredictionInput':
        if self.computed_at >= self.game_context.date_utc:
            raise ValueError("Temporal leakage: computed_at must be before game_context.date_utc")
        return self

class ReasoningOutput(BaseModel):
    thesis: str
    probability: float = Field(ge=0, le=1)
    confidence_grade: Literal["A", "B", "C", "D", "F"]
    key_drivers: List[str]
    counter_signals: List[str]
    uncertainty_notes: List[str]

class AuditOutput(BaseModel):
    pass_fail: Literal["pass", "fail"]
    warnings: List[str]
    fixes: List[str]