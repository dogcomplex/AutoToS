from abc import ABC, abstractmethod
from typing import Any, List, Tuple

class BaseDomain(ABC):
    @property
    @abstractmethod
    def GAME_CONFIG(self) -> dict:
        pass

    @property
    @abstractmethod
    def GOAL_PROMPT(self) -> str:
        pass

    @property
    @abstractmethod
    def SUCCESSOR_PROMPT(self) -> str:
        pass

    @abstractmethod
    def test_goal_function(self, code: str, enable_unit_tests: bool) -> Tuple[bool, str]:
        pass

    @abstractmethod
    def test_successor_function(self, succ_code: str, goal_code: str, enable_unit_tests: bool) -> Tuple[bool, str]:
        pass

    @abstractmethod
    def generate_random_initial_state(self) -> Any:
        pass

    @abstractmethod
    def validate_solution(self, solution: List[Any]) -> bool:
        pass

    @abstractmethod
    def get_state_key(self, state: Any) -> Any:
        pass