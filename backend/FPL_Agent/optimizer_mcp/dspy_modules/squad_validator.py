from __future__ import annotations
from collections import Counter
from typing import List

import dspy
from ...schema import Constraints, Player

class Squad_Validator(dspy.Module):
    """Validate a squad against a set of constraints"""
    def forward(self, squad: List[Player], constraints: Constraints) -> dict:
        """Validate a squad against a set of constraints"""
        violations: List[str] = []

        total_cost = sum(player.price for player in squad)
        if total_cost>constraints.budget:
            violations.append(f"Total cost of squad ({total_cost}) exceeds budget ({constraints.budget})")

        required_total = 15 
        if len(squad) != required_total:
            violations.append(f"Squad must have {required_total} players, but has {len(squad)}")

        position_counts = Counter(player.position for player in squad)
        for position, required_count in constraints.positions.items():
            count = position_counts.get(position, 0)
            if count != required_count:
                violations.append(f"Squad must have {required_count} players in position {position}, but has {count}")

        club_counts = Counter(player.team for player in squad)
        for club, count in club_counts.items():
            required_count = constraints.max_per_club
            if count>required_count:
                violations.append(f"Too many players from club {club} ({count} > {required_count})")

        return {"valid": len(violations) == 0, "violations": violations}