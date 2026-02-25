from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    user_input : str
    plan : List[Dict[str, Any]]
    current_step : int
    data : Dict[str, Any]
    artifcats : Dict[str, str]
    errors : List[str]


