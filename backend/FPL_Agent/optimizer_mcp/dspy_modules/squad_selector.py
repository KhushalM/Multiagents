from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import dspy
from ...schema import Player, Constraints

def _group_by_position(players: List[Player]) -> Dict[str, List[Player]]:
    """Group players by position"""
    grouped: Dict[str, List[Player]] = defaultdict(list)
    for player in players:
        grouped[player.position].append(player)
    return grouped

def _fallback_fill(grouped: Dict[str, List[Player]], required_by_pos: Dict[str, int], excluded_ids: set[int], club_counts: Dict[str, int], max_per_club: int, remaining_budget: float) -> Tuple[List[Player], float]:
    """
    Fallback: fill remaining slots with cheapest viable players across all needed positions.
    Greedy by price to fit budget.
    """
    print(f"Fallback fill called with {len(grouped)} positions and {len(required_by_pos)} required positions")
    picks: List[Player] = []
    flat: List[Tuple[str, Player]] = []
    for pos, need in required_by_pos.items():
        if need <=0:
            continue
        cheapest = sorted([p for p in grouped.get(pos, []) if p.id not in excluded_ids],
        key=lambda p: (p.price, -p.points, p.name))

        for p in cheapest:
            flat.append((pos, p))
        
    flat.sort(key=lambda x: (x[1].price, -x[1].points, x[1].name))

    for pos, player in flat:
        if required_by_pos[pos] <= 0:
            continue
        if player.id in excluded_ids:
            continue
        if club_counts[player.team] >= max_per_club:
            continue
        if player.price > remaining_budget:
            continue
        
        picks.append(player)
        required_by_pos[pos] -= 1
        club_counts[player.team] += 1
        remaining_budget -= player.price
        excluded_ids.add(player.id)

    return picks, remaining_budget

class Squad_Selector(dspy.Module):
    """
    Greedy squad selection module.
    - Buckets players by position.
    - For each position, fill with most points-per-dollar players.
    - Enforce max 3 per club.
    - Keep under budget: If we get stuck, fill with cheapest players.
    """
    def forward(self, player_pool: List[Player], constraints: Constraints, budget: float, seed_names: Optional[List[str]] = None, prefer_points: bool = True) -> dict:
        """Select a squad from the player pool"""
        required_by_pos: Dict[str, int] = dict(constraints.positions)
        total_required = sum(required_by_pos.values())
        grouped = _group_by_position(player_pool)

        for pos in grouped:
            if prefer_points:
                grouped[pos].sort(key=lambda p: (-p.points, p.price, p.name))
            else:
                grouped[pos].sort(key=lambda p: (p.price, -p.points, p.name))

        squad: List[Player] = []
        club_counts: Dict[str, int] = defaultdict(int)
        remaining_budget = float(budget)
        picked_ids: set[int] = set()

        def find_best_match(name: str) -> Optional[Player]:
            q = name.lower().strip()
            matches = [player for player in player_pool if q in player.name.lower()]
            if not matches:
                return None
            return sorted(matches, key=lambda p: (-p.points, p.price, p.name))[0]
        
        if seed_names:
            for name in seed_names:
                player = find_best_match(name)
                if not player:
                    continue
                if player.id in picked_ids:
                    continue
                if player.price > remaining_budget:
                    continue
                if club_counts[player.team] >= constraints.max_per_club:
                    continue
                squad.append(player)
                required_by_pos[player.position] -= 1
                club_counts[player.team] += 1
                remaining_budget -= player.price
                picked_ids.add(player.id)
                need -= 1
        
        for pos, need in required_by_pos.items():
            candidates = grouped.get(pos, [])
            for player in candidates:
                if need <= 0:
                    break
                if player.id in picked_ids:
                    continue
                if player.price > remaining_budget:
                    continue
                if club_counts[player.team] >= constraints.max_per_club:
                    continue
                squad.append(player)
                required_by_pos[pos] -= 1
                club_counts[player.team] += 1
                remaining_budget -= player.price
                picked_ids.add(player.id)
                need -= 1
        
        if len(squad) < total_required:
            fallback_picks, remaining_budget = _fallback_fill(grouped, required_by_pos, picked_ids, club_counts, constraints.max_per_club, remaining_budget)
            squad.extend(fallback_picks)
        
        total_cost = sum(player.price for player in squad)

        return {
            "squad": squad,
            "budget_used": total_cost
        }