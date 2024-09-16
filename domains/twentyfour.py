from typing import List, Tuple, Any
from domains.base_domain import BaseDomain
import random

class TwentyFourDomain(BaseDomain):
    @property
    def GAME_CONFIG(self) -> dict:
        return {
            'INITIAL_STATES': [
                [6, 6, 6, 6]
            ],
            'NUM_RANDOM_STATES': 0,
        }

    @property
    def GOAL_PROMPT(self) -> str:
        return """
        Task: Write a Python function called 'isgoal(state)' that returns True if the state is a goal state in the 24 Game and False otherwise. A state is a goal state if it contains exactly one number, which is 24.

        Constraints:
        - Do not modify the input state.
        - Ensure that the function works for any valid state.
        - The function should return a boolean value (True or False).
        - Handle potential edge cases, such as empty lists or lists with non-numeric elements.

        Examples:
        isgoal([24]) -> True
        isgoal([24, 1]) -> False
        isgoal([23]) -> False
        isgoal([]) -> False
        isgoal([24.0]) -> True

        Provide the complete 'isgoal' function.
        """

    @property
    def SUCCESSOR_PROMPT(self) -> str:
        return """
        Task: Write a Python function called 'succ(state)' that generates all valid successor states for the 24 Game. The input 'state' is a list of numbers (e.g., [6, 4, 3, 2]), and the output should be a list of successor states, where each successor state is a list of numbers resulting from applying one arithmetic operation (+, -, *, /) to any two numbers in the state.

        Constraints:
        - Use each pair of numbers only once per operation.
        - Do not generate duplicate states.
        - Do not modify the input state. Create new lists for successor states.
        - Each operation combines two numbers into one, reducing the state size by one.
        - Handle potential division by zero cases.
        - Ensure that each successor state has exactly one fewer element than the input state.

        Examples:
        succ([6, 4]) -> [[10], [2], [24], [1.5]]
        succ([1, 2, 3]) -> [[3, 3], [1, 5], [1, 1], [1, 6], [-1, 3], [1, 0.5]]

        Provide the complete 'succ' function. Ensure that you're creating new lists for successor states and not modifying the input state.
        """

    @property
    def GOAL_UNIT_TESTS(self):
        return [
            {'input': [24], 'expected': True},
            {'input': [24, 1], 'expected': False},
            {'input': [23], 'expected': False},
            {'input': [], 'expected': False},
            {'input': [24.0], 'expected': True},
        ]

    @property
    def SUCCESSOR_UNIT_TESTS(self):
        return SUCCESSOR_UNIT_TESTS  # This refers to the global variable defined at the bottom of the file

    def test_goal_function(self, code: str, enable_unit_tests: bool) -> Tuple[bool, str]:
        # Create a local scope to execute the code
        local_scope = {}
        try:
            exec(code, {}, local_scope)
            isgoal = local_scope['isgoal']
        except Exception as e:
            return False, f"Exception occurred when defining isgoal: {e}"

        if not enable_unit_tests:
            return True, "Unit tests disabled"

        for test in self.GOAL_UNIT_TESTS:
            input_state = test['input']
            expected = test['expected']
            try:
                result = isgoal(input_state.copy())
            except Exception as e:
                return False, f"Exception occurred when executing isgoal: {e}"
            if result != expected:
                return False, f"Test failed for input {input_state}: expected {expected}, got {result}"
        return True, "All tests passed"

    def test_successor_function(self, succ_code: str, goal_code: str, enable_unit_tests: bool) -> Tuple[bool, str]:
        # Create a local scope to execute the code
        local_scope = {}
        try:
            exec(succ_code, {}, local_scope)
            exec(goal_code, {}, local_scope)
            succ = local_scope['succ']
            isgoal = local_scope['isgoal']
        except Exception as e:
            return False, f"Exception occurred when defining functions: {e}"

        if not enable_unit_tests:
            return True, "Unit tests disabled"

        # Run unit tests
        for test in SUCCESSOR_UNIT_TESTS:
            current_state = test['input']
            expected_successors = test['expected_successors']
            try:
                successors = succ(current_state.copy())
            except Exception as e:
                return False, f"Exception occurred when executing succ: {e}"

            # Check if the number of successors matches
            if len(successors) != len(expected_successors):
                return False, f"Incorrect number of successors for {current_state}. Expected {len(expected_successors)}, got {len(successors)}"

            # Check if all expected successors are present
            for expected in expected_successors:
                if expected not in successors:
                    return False, f"Missing expected successor {expected} for {current_state}"

            # Check if there are any unexpected successors
            for successor in successors:
                if successor not in expected_successors:
                    return False, f"Unexpected successor {successor} for {current_state}"

            # Validate transition
            for successor in successors:
                valid, feedback = self.validate_transition(current_state, successor)
                if not valid:
                    return False, feedback

        return True, "All tests passed"

    def generate_random_initial_state(self) -> List[int]:
        return [random.randint(1, 13) for _ in range(4)]

    def validate_solution(self, solution: List[List[int]]) -> bool:
        if solution is None:
            return False
        # Check if the last state is the goal state
        return solution[-1] == [24]

    def get_state_key(self, state: List[int]) -> Tuple[int, ...]:
        return tuple(sorted(state))

    def validate_transition(self, s: List[int], t: List[int]) -> Tuple[bool, str]:
        if len(s) - len(t) != 1:
            feedback = f"Invalid transformation: length mismatch - the length of a successor must be one less than the parent. Parent: {s}, Successor: {t}"
            feedback += "\nLet's think step by step. First think through in words why the successor function produced a successor that had a length that was not exactly one less than the parent. Then provide the complete Python code for the revised successor function that ensures the length of a successor is exactly one less than the parent."
            feedback += "\nRemember how you fixed the previous mistakes, if any. Keep the same function signature."
            return False, feedback
        return True, ""

SUCCESSOR_UNIT_TESTS = [
    {
        'input': [1, 1, 4, 6],
        'expected_successors': [[1, 1, 10], [0.6666666666666666, 1, 1], [1, 4, 7], [-2, 1, 1], [-5, 1, 4], [1, 4, 6], [1, 1, 2], [1, 5, 6], [0.25, 1, 6], [-3, 1, 6], [0, 4, 6], [0.16666666666666666, 1, 4], [1, 1, 24], [1, 3, 6], [2, 4, 6], [1, 4, 5], [1, 1, 1.5]],
    },
    {
        'input': [1, 1, 11, 11],
        'expected_successors': [[1, 11, 11], [0.09090909090909091, 1, 11], [0, 11, 11], [1, 1, 22], [2, 11, 11], [0, 1, 1], [1, 1, 121], [1, 11, 12], [1, 1, 1.0], [1, 10, 11], [-10, 1, 11]],
    },
    {
        'input': [1, 1, 3, 8],
        'expected_successors': [[-2, 1, 8], [1, 3, 8], [0.3333333333333333, 1, 8], [-7, 1, 3], [1, 1, 2.6666666666666665], [0.125, 1, 3], [2, 3, 8], [1, 3, 7], [1, 1, 11], [1, 1, 5], [1, 1, 24], [-5, 1, 1], [0.375, 1, 1], [1, 2, 8], [1, 3, 9], [0, 3, 8], [1, 4, 8]],
    },
    {
        'input': [1, 1, 1, 8],
        'expected_successors': [[0.125, 1, 1], [1, 1, 9], [1, 1, 8], [0, 1, 8], [1, 2, 8], [1, 1, 7], [-7, 1, 1]],
    },
    {
        'input': [6, 6, 6, 6],
        'expected_successors': [[1.0, 6, 6], [6, 6, 12], [0, 6, 6], [6, 6, 36]],
    },
    {
        'input': [1, 1, 2, 12],
        'expected_successors': [[1, 3, 12], [-10, 1, 1], [1, 1, 10], [2, 2, 12], [1, 2, 13], [0.5, 1, 12], [-11, 1, 2], [1, 1, 12], [1, 1, 6.0], [1, 2, 12], [0, 2, 12], [1, 1, 24], [1, 2, 11], [1, 1, 14], [0.16666666666666666, 1, 1], [0.08333333333333333, 1, 2], [-1, 1, 12]],
    },
    {
        'input': [1, 2, 2, 6],
        'expected_successors': [[2, 2, 6], [-5, 2, 2], [2, 3, 6], [1, 2, 6], [0.3333333333333333, 1, 2], [2, 2, 5], [1, 1.0, 6], [0.16666666666666666, 2, 2], [1, 4, 6], [0, 1, 6], [-4, 1, 2], [1, 2, 12], [1, 2, 3.0], [2, 2, 7], [-1, 2, 6], [1, 2, 8], [1, 2, 4], [0.5, 2, 6]],
    },
    {
        'input': [1, 1, 10, 12],
        'expected_successors': [[-9, 1, 12], [1, 1, 1.2], [0.08333333333333333, 1, 10], [-2, 1, 1], [1, 10, 13], [1, 1, 22], [2, 10, 12], [0.1, 1, 12], [1, 1, 120], [1, 1, 2], [0.8333333333333334, 1, 1], [1, 9, 12], [1, 10, 12], [0, 10, 12], [1, 11, 12], [1, 10, 11], [-11, 1, 10]],
    },
    {
        'input': [2, 2, 10, 10],
        'expected_successors': [[0, 2, 2], [2, 10, 12], [1.0, 10, 10], [0, 10, 10], [-8, 2, 10], [2, 2, 100], [2, 5.0, 10], [1.0, 2, 2], [2, 2, 20], [4, 10, 10], [0.2, 2, 10], [2, 8, 10], [2, 10, 20]],
    },
    {
        'input': [1, 1, 1, 12],
        'expected_successors': [[0.08333333333333333, 1, 1], [1, 1, 13], [1, 1, 12], [0, 1, 12], [1, 2, 12], [-11, 1, 1], [1, 1, 11]],
    },
    {
        'input': [1, 4, 6],
        'expected_successors': [[4, 6]],
    },
    {
        'input': [4, 6],
        'expected_successors': [[24]],
    },
    {
        'input': [1, 1, 22],
        'expected_successors': [[1, 23]],
    },
    {
        'input': [1, 23],
        'expected_successors': [[24]],
    },
    {
        'input': [1, 1, 24],
        'expected_successors': [[1, 24]],
    },
    {
        'input': [1, 24],
        'expected_successors': [[24]],
    },
    {
        'input': [1, 2, 8],
        'expected_successors': [[3, 8]],
    },
    {
        'input': [3, 8],
        'expected_successors': [[24]],
    },
    {
        'input': [6, 6, 12],
        'expected_successors': [[6, 18]],
    },
    {
        'input': [6, 18],
        'expected_successors': [[24]],
    },
    {
        'input': [0, 2, 12],
        'expected_successors': [[0, 24]],
    },
    {
        'input': [0, 24],
        'expected_successors': [[24]],
    },
    {
        'input': [2, 2, 6],
        'expected_successors': [[2, 12]],
    },
    {
        'input': [2, 12],
        'expected_successors': [[24]],
    },
    {
        'input': [1, 11, 12],
        'expected_successors': [[11, 13]],
    },
    {
        'input': [11, 13],
        'expected_successors': [[24]],
    },
    {
        'input': [2, 2, 20],
        'expected_successors': [[2, 22]],
    },
    {
        'input': [2, 22],
        'expected_successors': [[24]],
    },
    {
        'input': [1, 1, 12],
        'expected_successors': [[2, 12]],
    },
]