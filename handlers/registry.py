import logging
from typing import Dict
from handlers.base_handler import BaseHandler

logger = logging.getLogger(__name__)

<<<<<<< HEAD

=======
>>>>>>> 02f35c4f7760554f44d25842038d030ffde2861c
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


<<<<<<< HEAD
# Singleton
registry = HandlerRegistry()


def load_all_handlers() -> None:
    """Import all handlers so they self-register. Call once at startup."""
=======
registry = HandlerRegistry()


def load_all_handler() -> None:
>>>>>>> 02f35c4f7760554f44d25842038d030ffde2861c
    import handlers.database_handler     # noqa
    import handlers.script_handler       # noqa
    import handlers.api_handler          # noqa
    import handlers.notification_handler # noqa
    import handlers.file_handler         # noqa
<<<<<<< HEAD
    logger.info("[Registry] Loaded handlers: %s", registry.available())
=======
    logger.info("[Registry] Loaded handlers: %s", registry.available())













>>>>>>> 02f35c4f7760554f44d25842038d030ffde2861c
