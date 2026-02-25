def planner_node(state):
    """
    Hardcoded planner for now
    """
    plan = [
        {"type": "validate", "payload": {}},
        {"type": "sql", "payload": {"query": "SELECT * FROM sales"}},
        {"type": "script", "payload": {"script": "generate_report.py"}},
        {"type": "file", "payload": {"output": "report.csv"}}
    ]

    state["plan"] = plan
    state["current_step"] = 0
    return state




