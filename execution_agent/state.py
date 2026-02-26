from typing import TypedDict, List, Optional, Dict

class WorkflowState(TypedDict):
    workflow : Dict[str, any]
    current_step_index: List[str]
    files_created = List[str]
    logs : List[str]
    status : str

    last_step_output : Optional[str]
    error : Optional[str]
    

