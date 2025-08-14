from typing import List, Dict, Any
from pydantic import BaseModel

class Constraints(BaseModel):
    """Constraints for the FPL Agent"""
    budget: float = 100.0
    positions: Dict[str, int] = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    max_per_club: int = 3

class Player(BaseModel):
    """Player schema"""
    id: int
    name: str
    position: str #GK, DEF, MID, FWD
    team: str
    price: float
    points: int = 0

class ProposeRequest(BaseModel):
    """Propose request schema"""
    player_pool: List[Player]
    constraints: Constraints
    budget: float = 100.0

class ProposeResponse(BaseModel):
    """Propose response schema"""
    squad: List[Player]
    constraints: Constraints

class ValidateRequest(BaseModel):
    """Validate request schema"""
    squad: List[Player]
    constraints: Constraints

class ValidateResponse(BaseModel):
    """Validate response schema"""
    valid: bool
    violations: List[str]