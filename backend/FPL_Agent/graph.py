from typing import TypedDict, List
from langgraph.graph import END, START, StateGraph
from .schema import Constraints, Player
from .fpl_data_client import get_player_pool
from .optimizer_mcp.dspy_modules.squad_selector import Squad_Selector
from .optimizer_mcp.dspy_modules.squad_validator import Squad_Validator
from langchain_openai import ChatOpenAI
import json

class State(TypedDict, total=False):
    pool: List[Player]
    constraints: Constraints
    budget: float
    squad: List[Player]
    total_cost: float
    violations: List[str]
    explanation: str
    seed_names: List[str]

def fetch_data(state: State) -> State:
    pool = get_player_pool()
    return {**state, "pool": pool}

def propose_squad(state: State) -> State:
    selector = Squad_Selector()
    res = selector(player_pool = state["pool"], constraints = state["constraints"], budget = state["budget"], seed_names = state["seed_names"], prefer_points = True)
    squad = res["squad"]
    total_cost = res.get("total_cost", res.get("budget_used", sum(p.price for p in squad)))
    return {**state, "squad" : squad, "total_cost" : total_cost}

def validate_squad(state:State) -> State:
    validator = Squad_Validator()
    verdict = validator(squad = state["squad"], constraints = state["constraints"])
    return {**state, "violations" : verdict["violations"]}

def is_valid(state:State) -> State:
    return not state.get("violations")

def repair_squad(state:State) -> State:
    #v0.1 simple retry.
    selector = Squad_Selector()
    res = selector(player_pool = state["pool"], constraints = state["constraints"], budget = state["budget"])
    squad = res["squad"]
    total_cost = res.get("total_cost", res.get("budget_used", sum(p.price for p in squad)))
    return {**state, "squad" : squad, "total_cost" : float(total_cost), "violations" : []}

def llm_plan(state:State) -> State:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = (
        "You are helping plan an FPL squad.\n"
        f"From the pool of players available: {state['pool']} and following constraints: {state['constraints']}, \n"
        "Come up with a solid squad that is under budget and satisfies the constraints while also having the best players."
    )
    msg = llm.invoke(prompt)
    content = msg.content if hasattr(msg, "content") else ""
    try:
        data = json.loads(content)
        seed_names = data.get("seed_names", [])
        if not isinstance(seed_names, list):
            seed_names = []
    except Exception:
        seed_names = []
    return {**state, "seed_names" : seed_names}

def explain_squad(state: State) -> State:
    if state.get("violations"):
        return state
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    lines = [f"{p.name} - {p.position} - {p.team} - £{p.price}" for p in state["squad"]]
    prompt = (
        "You are helping explain an FPL squad.\n"
        f"Budget used: £{state['total_cost']:.1f}m\n"
        "Squad:\n" + "\n".join(lines) + "\n"
        "Briefly justify the selection focus (budget vs points vs balance) in 3-5 sentences."
        "--------------------------------\n"
        "Improvements:\n"
        f"From the pool of players available: {state['pool']} and following constraints: {state['constraints']}, \n"
        "Explain ways to improve the squad."
        "Suggest me a better squad if you can after suggesting improvements. Leaving only 0.5m to spend."
    )
    msg = llm.invoke(prompt)
    return {**state, "explanation" : msg.content if hasattr(msg, "content") else str(msg)}

def build_app():
    g = StateGraph(State)
    g.add_node("fetch_data", fetch_data)
    g.add_node("propose_squad", propose_squad)
    g.add_node("validate_squad", validate_squad)
    g.add_node("repair_squad", repair_squad)
    g.add_node("llm_plan", llm_plan)
    g.set_entry_point("fetch_data")
    print("Added fetch_data")
    g.add_edge("fetch_data", "llm_plan")
    print("Added llm_plan")
    g.add_edge("llm_plan", "propose_squad")
    print("Added propose_squad")
    g.add_edge("propose_squad", "validate_squad")
    print("Added validate_squad")
    g.add_node("explain_squad", explain_squad)
    g.add_conditional_edges("validate_squad", lambda s: "ok" if is_valid(s) else "repair", {
        "ok" : "explain_squad",
        "repair" : "repair_squad"
    })
    g.add_edge("explain_squad", END)
    return g.compile()

def main():
    app = build_app()
    inital = {"constraints" : Constraints(), "budget" : 100.0}
    final = app.invoke(inital)
    print("Valid", not final.get("violations"))
    if final.get("violations"):
        print("Violations", final["violations"])
    else:
        print(f"Squad Size: {len(final['squad'])} and Total Cost: {final['total_cost']}")
        for player in final["squad"]:
            print(f"{player.name} - {player.position} - {player.team} - £{player.price}")
        print("Explanation: ", final.get("explanation"))

if __name__ == "__main__":
    main()