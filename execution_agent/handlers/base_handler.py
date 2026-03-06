from abc import ABC, abstractmethod

class BaseHandler(ABC):
    @property
    @abstractmethod

    def name(self) -> str:
        """Step type string this handler handles e.g. 'database_read'"""

    @abstractmethod
    def execute(self, step: dict, state: dict) -> dict:
        """Execute step, return updated state."""


