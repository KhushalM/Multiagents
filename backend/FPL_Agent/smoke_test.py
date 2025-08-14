from .fpl_data_client import get_player_pool
from .schema import Constraints, ProposeRequest, ProposeResponse, ValidateRequest, ValidateResponse
from .optimizer_mcp.dspy_modules.squad_validator import Squad_Validator
from .optimizer_mcp.dspy_modules.squad_selector import Squad_Selector

def main():
    pool = get_player_pool()
    print(f"Loaded {len(pool)} players")

    constraints = Constraints()
    selector = Squad_Selector()
    validator = Squad_Validator()

    selected = selector(pool, constraints, 100.0)
    squad = selected["squad"]
    total_cost = selected["budget_used"]
    print(f"Selected squad: {squad}")
    print(f"Total cost: {total_cost}")

    valid = validator(squad, constraints)
    print(f"Validation result: {valid}")
    if not valid["valid"]:
        print("Violations:")
        for violation in valid["violations"]:
            print(f"- {violation}")

if __name__ == "__main__":
    main()