import openai
import time
import threading
from openai import OpenAI
import re
import os
from dotenv import load_dotenv
from collections import deque
from typing import List, Callable, Any

from envs.twentyfour import (
    GAME_CONFIG, GOAL_UNIT_TESTS, SUCCESSOR_UNIT_TESTS,
    generate_random_initial_state, validate_transition_complex,
    validate_solution, SUCCESSOR_PROMPT, GOAL_PROMPT
)

# Load environment variables from .env file
load_dotenv()

# Configuration section
CONFIG = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'MODEL_NAME': 'gpt-4',
    'MAX_ITERATIONS': 100,
    'FUNCTION_TIMEOUT': 1,
}

# Set OpenAI API key
openai.api_key = CONFIG['OPENAI_API_KEY']

# Define the system prompt as per the paper
SYSTEM_PROMPT = """
You are a Python coding assistant. Help me generate my Python functions based on the task descriptions. Please always generate only a single function and keep all imports in it. If you need to define any additional functions, define them as inner functions. Do not generate examples of how to invoke the function. Please do not add any print statements outside the function. Provide the complete function and do not include any ellipsis notation.
"""

# Global counters for function calls and LLM queries
function_calls = {
    'get_code_from_api': 0,
    'test_goal_function': 0,
    'test_successor_function': 0,
    'bfs_search': 0
}

llm_queries = 0
total_llm_tokens = 0

def execute_with_timeout(func: Callable, args: tuple = (), timeout: int = CONFIG['FUNCTION_TIMEOUT']) -> Any:
    result = [TimeoutError("Function execution timed out")]
    def target():
        try:
            result[0] = func(*args)
        except Exception as e:
            result[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return TimeoutError("Function execution timed out")
    return result[0]

def get_code_from_api(prompt: str) -> str:
    global llm_queries, total_llm_tokens
    llm_queries += 1
    function_calls['get_code_from_api'] += 1
    
    client = OpenAI(api_key=openai.api_key)
    start_time = time.time()
    response = client.chat.completions.create(
        model=CONFIG['MODEL_NAME'],
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0,
    )
    end_time = time.time()
    
    total_llm_tokens += response.usage.total_tokens
    
    content = response.choices[0].message.content
    
    # Extract Python code from the response
    code_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        code = content.strip()
    
    print(f"LLM query took {end_time - start_time:.2f} seconds")
    return code

def test_goal_function(code: str) -> tuple[bool, str]:
    function_calls['test_goal_function'] += 1
    # Create a local scope to execute the code
    local_scope = {}
    try:
        exec(code, {}, local_scope)
        isgoal = local_scope['isgoal']
    except Exception as e:
        return False, f"Exception occurred when defining isgoal: {e}"

    # Run unit tests
    for test in GOAL_UNIT_TESTS:
        input_state = test['input']
        expected = test['expected']
        try:
            result = isgoal(input_state.copy())
        except Exception as e:
            return False, f"Exception occurred when executing isgoal: {e}"
        if result != expected:
            return False, f"Test failed for input {input_state}: expected {expected}, got {result}"
    return True, "All tests passed"

def test_successor_function(succ_code: str, isgoal_code: str) -> tuple[bool, str]:
    function_calls['test_successor_function'] += 1
    # Create a local scope to execute the code
    local_scope = {}
    try:
        exec(succ_code, {}, local_scope)
        exec(isgoal_code, {}, local_scope)
        succ = local_scope['succ']
        isgoal = local_scope['isgoal']
    except Exception as e:
        return False, f"Exception occurred when defining functions: {e}"

    # Partial soundness check during BFS
    initial_state = GAME_CONFIG['INITIAL_STATES'][0]  # Example initial state
    goal_found = False
    visited = set()
    queue = deque()
    queue.append(tuple(initial_state))

    while queue:
        current_state = list(queue.popleft())
        original_state = current_state.copy()
        state_key = tuple(sorted(current_state))
        if state_key in visited:
            continue
        visited.add(state_key)

        # Goal test with timeout
        result = execute_with_timeout(isgoal, args=(current_state,))
        if isinstance(result, TimeoutError):
            return False, f"isgoal function timed out on input {current_state}"
        elif result:
            goal_found = True
            break

        # Successor generation with timeout
        result = execute_with_timeout(succ, args=(current_state,))
        if isinstance(result, TimeoutError):
            return False, f"succ function timed out on input {current_state}"
        elif isinstance(result, Exception):
            return False, f"Exception in succ function on input {current_state}: {result}"

        successors = result

        # Check for input state modification
        if current_state != original_state:
            return False, f"succ function modified the input state {original_state}"

        for successor in successors:
            # Partial soundness test
            valid, feedback = validate_transition_complex(current_state, successor)
            if not valid:
                return False, feedback
            queue.append(tuple(successor))

    if not goal_found:
        return False, "Failed to find a goal state during BFS"
    return True, "Successor function passed all tests"

def refine_goal_function() -> str:
    goal_code = ''
    iterations = 0
    max_iterations = CONFIG['MAX_ITERATIONS']
    while iterations < max_iterations:
        if iterations == 0:
            prompt = GOAL_PROMPT
        else:
            prompt = feedback_prompt
        goal_code = get_code_from_api(prompt)
        print(f"Received code:\n{goal_code}")  # Print the received code
        try:
            passed, message = test_goal_function(goal_code)
            if passed:
                print("Goal function passed all tests.")
                return goal_code
            else:
                print(f"Goal function failed: {message}")
                feedback_prompt = f"""
The goal test function failed with the following error: {message}
Please think about why this error occurred and provide a corrected version of the 'isgoal' function.
Remember to keep the same function signature and constraints.
Here's the previous attempt:
{goal_code}
"""
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Problematic code:\n{goal_code}")
        iterations += 1
    raise Exception("Failed to generate a correct goal function after multiple iterations.")

def refine_successor_function(goal_code: str) -> str:
    succ_code = ''
    iterations = 0
    max_iterations = CONFIG['MAX_ITERATIONS']
    while iterations < max_iterations:
        if iterations == 0:
            prompt = SUCCESSOR_PROMPT
        else:
            prompt = feedback_prompt
        succ_code = get_code_from_api(prompt)
        print(f"Received successor code:\n{succ_code}")  # Print the received code
        try:
            passed, message = test_successor_function(succ_code, goal_code)
            if passed:
                print("Successor function passed all tests.")
                return succ_code
            else:
                print(f"Successor function failed: {message}")
                if "modified the input state" in message:
                    print(f"Original state: {eval(message.split('state ')[-1])}")
                feedback_prompt = f"""
The successor function failed with the following error: {message}
Please think about why this error occurred and provide a corrected version of the 'succ' function.
Remember to keep the same function signature and constraints.
Here's the previous attempt:
{succ_code}
"""
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Problematic successor code:\n{succ_code}")
        iterations += 1
    raise Exception("Failed to generate a correct successor function after multiple iterations.")

def bfs_search(succ_code: str, isgoal_code: str, initial_state: List[int]) -> List[List[int]]:
    function_calls['bfs_search'] += 1
    # Create a local scope to execute the code
    local_scope = {}
    exec(succ_code, {}, local_scope)
    exec(isgoal_code, {}, local_scope)
    succ = local_scope['succ']
    isgoal = local_scope['isgoal']

    visited = set()
    queue = deque()
    queue.append((initial_state, []))  # (state, path)

    while queue:
        current_state, path = queue.popleft()
        state_key = tuple(sorted(current_state))
        if state_key in visited:
            continue
        visited.add(state_key)

        if isgoal(current_state):
            return path + [current_state]

        successors = succ(current_state)
        for successor in successors:
            queue.append((successor, path + [current_state]))
    return None  # No solution found

def main():
    start_time = time.time()
    
    print("Refining goal function...")
    goal_code = refine_goal_function()
    print("Refining successor function...")
    succ_code = refine_successor_function(goal_code)

    # Use predefined initial states
    for i, initial_state in enumerate(GAME_CONFIG['INITIAL_STATES'], 1):
        print(f"\nSolving for initial state {i}: {initial_state}")
        solution = bfs_search(succ_code, goal_code, initial_state)
        if validate_solution(solution):
            print("Solution found:")
            for state in solution:
                print(state)
        else:
            print("No solution found or solution is invalid.")

    # Generate and use random initial states
    for i in range(GAME_CONFIG['NUM_RANDOM_STATES']):
        initial_state = generate_random_initial_state()
        print(f"\nSolving for random initial state {i+1}: {initial_state}")
        solution = bfs_search(succ_code, goal_code, initial_state)
        if validate_solution(solution):
            print("Solution found:")
            for state in solution:
                print(state)
        else:
            print("No solution found or solution is invalid.")

    end_time = time.time()
    total_time = end_time - start_time

    print("\nSummary Statistics:")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"LLM model used: {CONFIG['MODEL_NAME']}")
    print(f"Total LLM queries: {llm_queries}")
    print(f"Total tokens used: {total_llm_tokens}")
    print("\nFunction call counts:")
    for func, count in function_calls.items():
        print(f"  {func}: {count}")

    print("\nOther important parameters:")
    print(f"Initial states: {GAME_CONFIG['INITIAL_STATES']}")
    print(f"Number of random states: {GAME_CONFIG['NUM_RANDOM_STATES']}")
    print(f"Max iterations for refining functions: {CONFIG['MAX_ITERATIONS']}")
    print(f"Timeout for function execution: {CONFIG['FUNCTION_TIMEOUT']} second")

if __name__ == '__main__':
    main()

def test_successor_function(succ_code, isgoal_code):
    function_calls['test_successor_function'] += 1
    # Create a local scope to execute the code
    local_scope = {}
    try:
        exec(succ_code, {}, local_scope)
        exec(isgoal_code, {}, local_scope)
        succ = local_scope['succ']
        isgoal = local_scope['isgoal']
    except Exception as e:
        return False, f"Exception occurred when defining functions: {e}"

    # Partial soundness check during BFS
    initial_state = GAME_CONFIG['INITIAL_STATES'][0]  # Example initial state
    goal_found = False
    visited = set()
    queue = deque()
    queue.append(tuple(initial_state))

    while queue:
        current_state = list(queue.popleft())
        original_state = current_state.copy()
        state_key = tuple(sorted(current_state))
        if state_key in visited:
            continue
        visited.add(state_key)

        # Goal test with timeout
        result = execute_with_timeout(isgoal, args=(current_state,))
        if isinstance(result, TimeoutError):
            return False, f"isgoal function timed out on input {current_state}"
        elif result:
            goal_found = True
            break

        # Successor generation with timeout
        result = execute_with_timeout(succ, args=(current_state,))
        if isinstance(result, TimeoutError):
            return False, f"succ function timed out on input {current_state}"
        elif isinstance(result, Exception):
            return False, f"Exception in succ function on input {current_state}: {result}"

        successors = result

        # Check for input state modification
        if current_state != original_state:
            return False, f"succ function modified the input state {original_state}"

        for successor in successors:
            # Partial soundness test
            valid, feedback = validate_transition_complex(current_state, successor)
            if not valid:
                return False, feedback
            queue.append(tuple(successor))

    if not goal_found:
        return False, "Failed to find a goal state during BFS"
    return True, "Successor function passed all tests"

def refine_goal_function() -> str:
    goal_code = ''
    iterations = 0
    max_iterations = CONFIG['MAX_ITERATIONS']
    while iterations < max_iterations:
        if iterations == 0:
            prompt = GOAL_PROMPT
        else:
            prompt = feedback_prompt
        goal_code = get_code_from_api(prompt)
        print(f"Received code:\n{goal_code}")  # Print the received code
        try:
            passed, message = test_goal_function(goal_code)
            if passed:
                print("Goal function passed all tests.")
                return goal_code
            else:
                print(f"Goal function failed: {message}")
                feedback_prompt = f"""
The goal test function failed with the following error: {message}
Please think about why this error occurred and provide a corrected version of the 'isgoal' function.
Remember to keep the same function signature and constraints.
Here's the previous attempt:
{goal_code}
"""
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Problematic code:\n{goal_code}")
        iterations += 1
    raise Exception("Failed to generate a correct goal function after multiple iterations.")

def refine_successor_function(goal_code: str) -> str:
    succ_code = ''
    iterations = 0
    max_iterations = CONFIG['MAX_ITERATIONS']
    while iterations < max_iterations:
        if iterations == 0:
            prompt = SUCCESSOR_PROMPT
        else:
            prompt = feedback_prompt
        succ_code = get_code_from_api(prompt)
        print(f"Received successor code:\n{succ_code}")  # Print the received code
        try:
            passed, message = test_successor_function(succ_code, goal_code)
            if passed:
                print("Successor function passed all tests.")
                return succ_code
            else:
                print(f"Successor function failed: {message}")
                if "modified the input state" in message:
                    print(f"Original state: {eval(message.split('state ')[-1])}")
                feedback_prompt = f"""
The successor function failed with the following error: {message}
Please think about why this error occurred and provide a corrected version of the 'succ' function.
Remember to keep the same function signature and constraints.
Here's the previous attempt:
{succ_code}
"""
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Problematic successor code:\n{succ_code}")
        iterations += 1
    raise Exception("Failed to generate a correct successor function after multiple iterations.")

def bfs_search(succ_code: str, isgoal_code: str, initial_state: List[int]) -> List[List[int]]:
    function_calls['bfs_search'] += 1
    # Create a local scope to execute the code
    local_scope = {}
    exec(succ_code, {}, local_scope)
    exec(isgoal_code, {}, local_scope)
    succ = local_scope['succ']
    isgoal = local_scope['isgoal']

    visited = set()
    queue = deque()
    queue.append((initial_state, []))  # (state, path)

    while queue:
        current_state, path = queue.popleft()
        state_key = tuple(sorted(current_state))
        if state_key in visited:
            continue
        visited.add(state_key)

        if isgoal(current_state):
            return path + [current_state]

        successors = succ(current_state)
        for successor in successors:
            queue.append((successor, path + [current_state]))
    return None  # No solution found

def main():
    start_time = time.time()
    
    print("Refining goal function...")
    goal_code = refine_goal_function()
    print("Refining successor function...")
    succ_code = refine_successor_function(goal_code)

    # Use predefined initial states
    for i, initial_state in enumerate(GAME_CONFIG['INITIAL_STATES'], 1):
        print(f"\nSolving for initial state {i}: {initial_state}")
        solution = bfs_search(succ_code, goal_code, initial_state)
        if validate_solution(solution):
            print("Solution found:")
            for state in solution:
                print(state)
        else:
            print("No solution found or solution is invalid.")

    # Generate and use random initial states
    for i in range(GAME_CONFIG['NUM_RANDOM_STATES']):
        initial_state = generate_random_initial_state()
        print(f"\nSolving for random initial state {i+1}: {initial_state}")
        solution = bfs_search(succ_code, goal_code, initial_state)
        if validate_solution(solution):
            print("Solution found:")
            for state in solution:
                print(state)
        else:
            print("No solution found or solution is invalid.")

    end_time = time.time()
    total_time = end_time - start_time

    print("\nSummary Statistics:")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"LLM model used: {CONFIG['MODEL_NAME']}")
    print(f"Total LLM queries: {llm_queries}")
    print(f"Total tokens used: {total_llm_tokens}")
    print("\nFunction call counts:")
    for func, count in function_calls.items():
        print(f"  {func}: {count}")

    print("\nOther important parameters:")
    print(f"Initial states: {GAME_CONFIG['INITIAL_STATES']}")
    print(f"Number of random states: {GAME_CONFIG['NUM_RANDOM_STATES']}")
    print(f"Max iterations for refining functions: {CONFIG['MAX_ITERATIONS']}")
    print(f"Timeout for function execution: {CONFIG['FUNCTION_TIMEOUT']} second")

if __name__ == '__main__':
    main()
