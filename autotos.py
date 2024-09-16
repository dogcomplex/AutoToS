# Filename: autotos_24game.py

import openai
import time
import copy
import threading
from openai import OpenAI
import re

# Configuration section
CONFIG = {
    'OPENAI_API_KEY': 'YOUR_API_KEY_HERE',
    'MODEL_NAME': 'gpt-3.5-turbo',
    'MAX_ITERATIONS': 100,
    'FUNCTION_TIMEOUT': 1,
    'INITIAL_STATE': [6, 6, 6, 6],
}

# Set OpenAI API key
openai.api_key = CONFIG['OPENAI_API_KEY']

# Define the system prompt as per the paper
SYSTEM_PROMPT = """
You are a Python coding assistant. Help me generate my Python functions based on the task descriptions. Please always generate only a single function and keep all imports in it. If you need to define any additional functions, define them as inner functions. Do not generate examples of how to invoke the function. Please do not add any print statements outside the function. Provide the complete function and do not include any ellipsis notation.
"""

# Initial prompt to generate the successor function
SUCCESSOR_PROMPT = """
Task: Write a Python function called 'succ(state)' that generates all valid successor states for the 24 Game. The input 'state' is a list of numbers (e.g., [6, 4, 3, 2]), and the output should be a list of successor states, where each successor state is a list of numbers resulting from applying one arithmetic operation (+, -, *, /) to any two numbers in the state.

Constraints:
- Use each pair of numbers only once per operation.
- Do not generate duplicate states.
- Do not modify the input state. Create new lists for successor states.
- Each operation combines two numbers into one, reducing the state size by one.

Example:
Input: [6, 4]
Output: [[10], [2], [24], [1.5]]

Note: 6 + 4 = 10, 6 - 4 = 2, 6 * 4 = 24, 6 / 4 = 1.5

Provide the complete 'succ' function. Ensure that you're creating new lists for successor states and not modifying the input state.
"""

# Initial prompt to generate the goal test function
GOAL_PROMPT = """
Task: Write a Python function called 'isgoal(state)' that returns True if the state is a goal state in the 24 Game and False otherwise. A state is a goal state if it contains exactly one number, which is 24.

Constraints:
- Do not modify the input state.
- Ensure that the function works for any valid state.

Example:
Input: [24]
Output: True

Input: [24, 1]
Output: False

Provide the complete 'isgoal' function.
"""

# Unit tests for the goal test function
GOAL_STATES = [
    [24],
]

NON_GOAL_STATES = [
    [],
    [3],
    [24, 1],
    [1, 6, 4],
    [1, 1, 4, 6],
]

# Update the GOAL_UNIT_TESTS to use these new variables
GOAL_UNIT_TESTS = (
    [{'input': state, 'expected': True} for state in GOAL_STATES] +
    [{'input': state, 'expected': False} for state in NON_GOAL_STATES]
)

# Successor completeness test data
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

# Partial soundness test: checks if successor state length is one less than parent state length
def validate_transition_complex(s, t):
    if len(s) - len(t) != 1:
        feedback = f"Invalid transformation: length mismatch - the length of a successor must be one less than the parent. Parent: {s}, Successor: {t}"
        feedback += "\nLet's think step by step. First think through in words why the successor function produced a successor that had a length that was not exactly one less than the parent. Then provide the complete Python code for the revised successor function that ensures the length of a successor is exactly one less than the parent."
        feedback += "\nRemember how you fixed the previous mistakes, if any. Keep the same function signature."
        return False, feedback
    return True, ""

# Function to execute with timeout
import threading

def execute_with_timeout(func, args=(), timeout=CONFIG['FUNCTION_TIMEOUT']):
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

# Global counters for function calls and LLM queries
function_calls = {
    'get_code_from_api': 0,
    'test_goal_function': 0,
    'test_successor_function': 0,
    'bfs_search': 0
}

llm_queries = 0
total_llm_tokens = 0

# Modify the get_code_from_api function to count LLM queries and tokens
def get_code_from_api(prompt):
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

# Modify other functions to increment their call counters
def test_goal_function(code):
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
            result = isgoal(copy.deepcopy(input_state))
        except Exception as e:
            return False, f"Exception occurred when executing isgoal: {e}"
        if result != expected:
            return False, f"Test failed for input {input_state}: expected {expected}, got {result}"
    return True, "All tests passed"

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
    from collections import deque

    initial_state = CONFIG['INITIAL_STATE']  # Example initial state
    goal_found = False
    visited = set()
    queue = deque()
    queue.append(tuple(initial_state))

    while queue:
        current_state = list(queue.popleft())
        original_state = copy.deepcopy(current_state)  # Make a deep copy
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

# Feedback loop for the goal function
def refine_goal_function():
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

# Feedback loop for the successor function
def refine_successor_function(goal_code):
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

# BFS search using the generated functions
def bfs_search(succ_code, isgoal_code, initial_state):
    function_calls['bfs_search'] += 1
    # Create a local scope to execute the code
    local_scope = {}
    exec(succ_code, {}, local_scope)
    exec(isgoal_code, {}, local_scope)
    succ = local_scope['succ']
    isgoal = local_scope['isgoal']

    from collections import deque

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

# Solution validation
def validate_solution(solution):
    if solution is None:
        return False
    # Check if the last state is the goal state
    return solution[-1] == [24]

# Modify the main function to print summary statistics
def main():
    start_time = time.time()
    
    print("Refining goal function...")
    goal_code = refine_goal_function()
    print("Refining successor function...")
    succ_code = refine_successor_function(goal_code)

    initial_state = CONFIG['INITIAL_STATE']
    print("Starting BFS search...")
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
    print(f"Initial state: {initial_state}")
    print(f"Max iterations for refining functions: {CONFIG['MAX_ITERATIONS']}")
    print(f"Timeout for function execution: {CONFIG['FUNCTION_TIMEOUT']} second")

if __name__ == '__main__':
    main()
