from typing import List, Tuple, Any, Dict
from domains.base_domain import BaseDomain
import random

class BlocksWorldDomain(BaseDomain):
    @property
    def GAME_CONFIG(self) -> dict:
        return {
            'NUM_BLOCKS': 4,
            'NUM_RANDOM_STATES': 5,
            'INITIAL_STATES': [
                {
                    "clear": ["b", "c"],
                    "on-table": ["a", "c"],
                    "arm-empty": True,
                    "holding": None,
                    "on": [["b", "a"]]
                },
                {
                    "clear": ["d"],
                    "on-table": ["a"],
                    "arm-empty": True,
                    "holding": None,
                    "on": [["b", "a"], ["c", "b"], ["d", "c"]]
                },
                {
                    "clear": ["a", "b", "c", "d"],
                    "on-table": ["a", "b", "c", "d"],
                    "arm-empty": True,
                    "holding": None,
                    "on": []
                },
                {
                    "clear": ["d"],
                    "on-table": ["a", "b"],
                    "arm-empty": True,
                    "holding": None,
                    "on": [["c", "a"], ["d", "c"]]
                },
                {
                    "clear": ["b", "d"],
                    "on-table": ["a", "c"],
                    "arm-empty": True,
                    "holding": None,
                    "on": [["b", "a"], ["d", "c"]]
                }
            ]
        }

    @property
    def GOAL_PROMPT(self) -> str:
        return """
        Task: Write a Python function called 'isgoal(state, goal)' that checks if the current state matches the goal state in the BlocksWorld domain. The function should return True if the current state satisfies the goal conditions, and False otherwise.

        Input parameters:
        - 'state': A dictionary representing the current state of the blocks
        - 'goal': A dictionary representing the goal conditions

        State and goal representation:
        - 'clear': list of blocks with no blocks on top of them
        - 'on-table': list of blocks directly on the table
        - 'arm-empty': boolean indicating if the robot arm is empty (only in state)
        - 'holding': block being held by the robot arm (only in state, None if arm is empty)
        - 'on': list of [block_a, block_b] pairs, where block_a is on top of block_b

        Note: The goal state may not specify all aspects of the state. It may only include the 'on' relationships that must be satisfied.

        Constraints:
        - Do not modify the input state or goal.
        - Ensure that the function works for any valid state and goal.

        Provide the complete 'isgoal' function.
        """

    @property
    def SUCCESSOR_PROMPT(self) -> str:
        return """
        Task: Write a Python function called 'succ(state)' that generates all valid successor states for the BlocksWorld domain. The input 'state' is a dictionary representing the current state of the blocks, and the output should be a list of successor states, where each successor state is a dictionary resulting from applying one valid action to the current state.

        State representation:
        - 'clear': list of blocks with no blocks on top of them
        - 'on-table': list of blocks directly on the table
        - 'arm-empty': boolean indicating if the robot arm is empty
        - 'holding': block being held by the robot arm (None if arm is empty)
        - 'on': list of [block_a, block_b] pairs, where block_a is on top of block_b

        Valid actions:
        1. Pick up a block from the table
        2. Put down a held block on the table
        3. Stack a held block on top of another block
        4. Unstack a block from on top of another block

        Constraints:
        - Do not modify the input state. Create new dictionaries for successor states.
        - Ensure that all actions are valid before generating a successor state.
        - Do not generate duplicate states.

        Provide the complete 'succ' function. Ensure that you're creating new dictionaries for successor states and not modifying the input state.
        """

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

        # Run unit tests
        for test in GOAL_UNIT_TESTS:
            state = test['input']
            goal = test['goal']
            expected = test['expected']
            try:
                result = isgoal(state.copy(), goal.copy())
            except Exception as e:
                return False, f"Exception occurred when executing isgoal: {e}"
            if result != expected:
                return False, f"Test failed for input state {state} and goal {goal}: expected {expected}, got {result}"
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

    def generate_random_initial_state(self) -> Dict[str, Any]:
        num_blocks = self.GAME_CONFIG['NUM_BLOCKS']
        blocks = [chr(ord('a') + i) for i in range(num_blocks)]
        random.shuffle(blocks)
        
        state = {
            "clear": [],
            "on-table": [],
            "arm-empty": True,
            "holding": None,
            "on": []
        }
        
        for i, block in enumerate(blocks):
            if i == 0:
                state["on-table"].append(block)
            else:
                state["on"].append([block, blocks[i-1]])
        
        state["clear"].append(blocks[-1])
        
        return state

    def validate_solution(self, solution: List[Dict[str, Any]]) -> bool:
        if not solution:
            return False
        
        for i in range(len(solution) - 1):
            valid, _ = self.validate_transition(solution[i], solution[i+1])
            if not valid:
                return False
        
        return True

    def get_state_key(self, state: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            tuple(sorted(state['clear'])),
            tuple(sorted(state['on-table'])),
            state['arm-empty'],
            state['holding'],
            tuple(sorted(tuple(sorted(on_pair)) for on_pair in state['on']))
        )

    def validate_transition(self, parent: Dict[str, Any], successor: Dict[str, Any]) -> Tuple[bool, str]:
        # Check if the number of clear blocks equals the number of on-table blocks
        if len(successor["clear"]) != len(successor["on-table"]):
            return False, "Number of clear blocks does not match number of on-table blocks"
        
        # Check if arm-empty and holding are consistent
        if successor["arm-empty"] == (successor["holding"] is not None):
            return False, "Inconsistent arm-empty and holding states"
        
        # Check if all blocks are accounted for
        all_blocks = set(successor["clear"] + successor["on-table"] + [b for a, b in successor["on"]])
        if successor["holding"]:
            all_blocks.add(successor["holding"])
        if len(all_blocks) != self.GAME_CONFIG['NUM_BLOCKS']:
            return False, "Not all blocks are accounted for"
        
        return True, ""

# Unit tests for the goal test function
GOAL_UNIT_TESTS = [
    {
        'input': {
            "clear": ["b"],
            "on-table": ["d"],
            "arm-empty": True,
            "holding": None,
            "on": [["a", "c"], ["b", "a"], ["c", "d"]]
        },
        'goal': {
            "on": [["a", "c"], ["b", "a"], ["c", "d"]]
        },
        'expected': True
    },
    {
        'input': {
            "clear": ["b", "c"],
            "on-table": ["a", "c"],
            "arm-empty": True,
            "holding": None,
            "on": [["b", "a"]]
        },
        'goal': {
            "on": [["b", "a"], ["c", "b"]]
        },
        'expected': False
    }
]

# Successor completeness test data
SUCCESSOR_UNIT_TESTS = [
    {
        'input': {
            "clear": ["b", "c"],
            "on-table": ["a", "c"],
            "arm-empty": True,
            "holding": None,
            "on": [["b", "a"]]
        },
        'expected_successors': [
            {
                "clear": ["a", "c"],
                "on-table": ["a", "c"],
                "arm-empty": False,
                "holding": "b",
                "on": []
            },
            {
                "clear": ["b", "c"],
                "on-table": ["a", "b", "c"],
                "arm-empty": True,
                "holding": None,
                "on": []
            },
            {
                "clear": ["b"],
                "on-table": ["a"],
                "arm-empty": False,
                "holding": "c",
                "on": [["b", "a"]]
            }
        ]
    }
]
