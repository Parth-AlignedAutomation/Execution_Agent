import logging
from typing import Dict
from execution_agent.handlers.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class HandlerRegistry:

    def __init__(self):
        self._handlers: Dict[str, BaseHandler] = {}

    def register(self, handler: BaseHandler) -> None:
        self._handlers[handler.name] = handler
        logger.debug("[Registry] Registered: '%s'", handler.name)

    def get(self, step_type: str) -> BaseHandler:
        if step_type not in self._handlers:
            raise ValueError(
                f"No handler for step type '{step_type}'. "
                f"Available: {list(self._handlers.keys())}"
            )
        return self._handlers[step_type]

    def available(self) -> list:
        return list(self._handlers.keys())


registry = HandlerRegistry()


def load_all_handler() -> None:

    import handlers.database_handler     # noqa
    import handlers.script_handler       # noqa
    import handlers.api_handler          # noqa
    import handlers.notification_handler # noqa
    import handlers.file_handler         # noqa


    logger.info("[Registry] Loaded handlers: %s", registry.available())













