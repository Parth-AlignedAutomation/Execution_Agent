def execution_router(state):
    step = state["plan"][state["current_step"]]

    step_type = step["type"]

    if step["type"] == "validate":
        return "validation"
    
    if step["type"] == "sql":
        return "db_executor"
    
    if step["type"] == "script":
        return "script_executor"
    
    if step["type"] == "file":
        return "file_ops"
    

    return "end"


    
    