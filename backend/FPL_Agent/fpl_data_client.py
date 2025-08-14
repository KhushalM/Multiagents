from typing import List
import requests
from .schema import Player

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD"
}

def get_player_pool() -> List[Player]:
    """Get the player pool from the FPL Data API"""
    response = requests.get(FPL_BOOTSTRAP_URL, timeout=30)
    response.raise_for_status()
    data = response.json()

    id_to_team = {team["id"]: team["name"] for team in data.get("teams", [])}

    player_pool: List[Player] = []
    for e in data.get("elements", []):
        position = POSITION_MAP.get(e.get("element_type"))
        team_name = id_to_team.get(e.get("team"))
        if position is None or team_name is None:
            continue
        player_pool.append(Player(
            id=e.get("id"),
            name=f'{e.get("first_name", "").strip()} {e.get("second_name", "").strip()}'.strip(),
            position=position,
            team=team_name,
            price=e.get("now_cost", 0) / 10,
            points=e.get("total_points", 0),
        ))
    return player_pool

if __name__ == "__main__":
    player_pool = get_player_pool()
    print(player_pool)