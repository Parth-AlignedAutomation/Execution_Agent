from typing import TypedDict, Dict, List, Optional, Any

class WorkflowState(TypedDict):
        workflow : Dict[str, Any]
        current_step_index : int
        files_created : List[str]
        logs : List[str]
        status : str
        last_step_output : Optional[str]
        error : Optional[str]


    
    
