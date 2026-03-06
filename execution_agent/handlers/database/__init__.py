from execution_agent.handlers.database import mongodb, mysql, bigquery, sqlite, postgres

_MODULES = [postgres, mongodb, mysql, bigquery, sqlite]
DB_ENGINES = {}

for _mod in _MODULES:
    _engine_name = _mod.__name__.split(".")[-1]
    DB_ENGINES[_engine_name] = _mod.ENGINE
    
    for _alias in getattr(_mod, "ALIASES", []):
        DB_ENGINES[_alias] = {"alias": _engine_name}


