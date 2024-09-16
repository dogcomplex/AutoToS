import openai
import time
import threading
from openai import OpenAI
import re
import os
from dotenv import load_dotenv
from typing import List, Callable, Any, Dict
import itertools
import json

from domains.base_domain import BaseDomain
from domains.twentyfour import TwentyFourDomain
from domains.blocksworld import BlocksWorldDomain

# Load environment variables from .env file
load_dotenv()

# Configuration section
CONFIG = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'MODEL_NAME': 'gpt-4o',
    'MAX_ITERATIONS': 5,
    'FUNCTION_TIMEOUT': 1,
    'DOMAIN': 'twentyfour',  # or 'blocksworld'
    'ENABLE_GOAL_UNIT_TESTS': True,
    'ENABLE_SUCCESSOR_UNIT_TESTS': False,
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
        thread.join()
        raise result[0]
    if isinstance(result[0], Exception):
        raise result[0]
    return result[0]

def get_code_from_api(prompt: str) -> str:
    global llm_queries, total_llm_tokens
    function_calls['get_code_from_api'] += 1
    llm_queries += 1

    client = OpenAI()
    response = client.chat.completions.create(
        model=CONFIG['MODEL_NAME'],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    total_llm_tokens += response.usage.total_tokens
    content = response.choices[0].message.content

    # Extract Python code from the response
    code_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        # Remove any existing import statements for itertools
        # code = re.sub(r'(from itertools import .*\n|import itertools\n)', '', code)
        return code
    else:
        # If no Python code block is found, return the entire content
        return content.strip()

def run_unit_tests(func, tests, is_successor_test=False, domain_name=''):
    results = {'passed': [], 'failed': []}
    for i, test in enumerate(tests, 1):
        try:
            if is_successor_test:
                result = func(test['input'].copy())
                expected = test['expected_successors']
                if compare_successors(result, expected):
                    results['passed'].append({
                        'test_num': i,
                        'input': test['input'],
                        'expected': expected,
                        'output': result
                    })
                else:
                    results['failed'].append({
                        'test_num': i,
                        'input': test['input'],
                        'expected': expected,
                        'output': result,
                        'message': f"Test {i} failed: Mismatch in successors"
                    })
            else:
                if domain_name == 'blocksworld':
                    result = func(test['input'].copy(), test['goal'].copy())
                else:
                    result = func(test['input'].copy())
                expected = test['expected']
                if result == expected:
                    results['passed'].append({
                        'test_num': i,
                        'input': test['input'],
                        'goal': test.get('goal', {}),
                        'expected': expected,
                        'output': result
                    })
                else:
                    results['failed'].append({
                        'test_num': i,
                        'input': test['input'],
                        'goal': test.get('goal', {}),
                        'expected': expected,
                        'output': result,
                        'message': f"Test {i} failed: Expected {expected}, got {result}"
                    })
        except Exception as e:
            results['failed'].append({
                'test_num': i,
                'input': test['input'],
                'goal': test.get('goal', {}),
                'message': f"Test {i} failed: Raised an exception: {str(e)}"
            })
    
    return results

def compare_successors(result, expected):
    if len(result) != len(expected):
        return False
    
    for r, e in zip(sorted(result), sorted(expected)):
        if len(r) != len(e):
            return False
        for r_val, e_val in zip(r, e):
            if isinstance(r_val, float) and isinstance(e_val, float):
                if abs(r_val - e_val) > 1e-6:  # Allow small floating-point differences
                    return False
            elif r_val != e_val:
                return False
    return True

def print_test_results(test_results):
    print("\nTest Results:")
    print(f"Passed tests: {len(test_results['passed'])}")
    print(f"Failed tests: {len(test_results['failed'])}")
    
    print("\nPassed Tests:")
    for result in test_results['passed']:
        print(f"  Test {result['test_num']}:")
        print(f"    Input: {result['input']}")
        if 'goal' in result:
            print(f"    Goal: {result['goal']}")
        print(f"    Expected: {result['expected']}")
        print(f"    Output: {result['output']}")
    
    print("\nFailed Tests:")
    for result in test_results['failed']:
        print(f"  Test {result['test_num']}:")
        print(f"    Input: {result['input']}")
        if 'goal' in result:
            print(f"    Goal: {result['goal']}")
        print(f"    Expected: {result['expected']}")
        print(f"    Output: {result['output']}")
        print(f"    Message: {result['message']}")

def refine_goal_function(domain: BaseDomain) -> str:
    goal_code = ''
    iterations = 0
    max_iterations = CONFIG['MAX_ITERATIONS']
    while iterations < max_iterations:
        if iterations == 0:
            prompt = domain.GOAL_PROMPT
        else:
            prompt = feedback_prompt
        goal_code = get_code_from_api(prompt)
        print(f"\nIteration {iterations + 1}: Received goal function code")
        print("Goal function:")
        print(goal_code)
        
        try:
            local_scope = {}
            exec(goal_code, {}, local_scope)
            isgoal = local_scope['isgoal']
            
            if CONFIG['ENABLE_GOAL_UNIT_TESTS']:
                test_results = run_unit_tests(isgoal, domain.GOAL_UNIT_TESTS)
                print_test_results(test_results)
                if test_results['failed']:
                    raise Exception("Some tests failed")
            print("Goal function passed all tests.")
            return goal_code
        except Exception as e:
            feedback_prompt = f"""
The goal test function passed {len(test_results['passed'])} tests and failed {len(test_results['failed'])} tests.

Passed tests:
{json.dumps(test_results['passed'], indent=2)}

Failed tests:
{json.dumps(test_results['failed'], indent=2)}

Please analyze both the passed and failed tests, and provide a corrected version of the 'isgoal' function.
Remember to keep the same function signature and constraints.
Here's the previous attempt:
{goal_code}
"""
        iterations += 1
    raise Exception("Failed to generate a correct goal function after multiple iterations.")

def refine_successor_function(domain: BaseDomain, goal_code: str) -> str:
    succ_code = ''
    iterations = 0
    max_iterations = CONFIG['MAX_ITERATIONS']
    while iterations < max_iterations:
        if iterations == 0:
            prompt = domain.SUCCESSOR_PROMPT
        else:
            prompt = feedback_prompt
        succ_code = get_code_from_api(prompt)
        print(f"\nIteration {iterations + 1}: Received successor function code")
        print("Successor function:")
        print(succ_code)
        
        try:
            local_scope = {}
            exec(succ_code, {}, local_scope)
            succ = local_scope['succ']
            
            if CONFIG['ENABLE_SUCCESSOR_UNIT_TESTS']:
                test_results = run_unit_tests(succ, domain.SUCCESSOR_UNIT_TESTS, is_successor_test=True)
                print_test_results(test_results)
                if test_results['failed']:
                    raise Exception("Some tests failed")
            print("Successor function passed all tests.")
            return succ_code
        except Exception as e:
            feedback_prompt = f"""
The successor function passed {len(test_results['passed'])} tests and failed {len(test_results['failed'])} tests.

Passed tests:
{json.dumps(test_results['passed'], indent=2)}

Failed tests:
{json.dumps(test_results['failed'], indent=2)}

Please analyze both the passed and failed tests, and provide a corrected version of the 'succ' function.
Remember to keep the same function signature and constraints.
Here's the previous attempt:
{succ_code}
"""
        iterations += 1
    raise Exception("Failed to generate a correct successor function after multiple iterations.")

def bfs_search(domain: BaseDomain, succ_code: str, isgoal_code: str, initial_state: Any) -> List[Any]:
    function_calls['bfs_search'] += 1
    local_scope = {}
    
    try:
        exec(succ_code, local_scope, local_scope)
        exec(isgoal_code, local_scope, local_scope)
        succ = local_scope['succ']
        isgoal = local_scope['isgoal']
    except Exception as e:
        print(f"Error in loading succ or isgoal functions: {e}")
        return None

    visited = set()
    queue = []
    queue.append((initial_state, []))

    while queue:
        current_state, path = queue.pop(0)
        state_key = domain.get_state_key(current_state)
        if state_key in visited:
            continue
        visited.add(state_key)

        try:
            if isgoal(current_state):
                return path + [current_state]

            successors = succ(current_state)
            for successor in successors:
                queue.append((successor, path + [current_state]))
        except Exception as e:
            print(f"Error during search: {e}")
            print(f"Problematic state: {current_state}")
            continue

    return None  # No solution found

def main():
    start_time = time.time()
    
    if CONFIG['DOMAIN'] == 'twentyfour':
        domain = TwentyFourDomain()
    elif CONFIG['DOMAIN'] == 'blocksworld':
        domain = BlocksWorldDomain()
    else:
        raise ValueError(f"Unknown domain: {CONFIG['DOMAIN']}")

    print("Refining goal function...")
    goal_code = refine_goal_function(domain)
    print("Refining successor function...")
    succ_code = refine_successor_function(domain, goal_code)

    # Use predefined initial states
    for i, initial_state in enumerate(domain.GAME_CONFIG['INITIAL_STATES'], 1):
        print(f"\nSolving for initial state {i}: {initial_state}")
        solution = bfs_search(domain, succ_code, goal_code, initial_state)
        if domain.validate_solution(solution):
            print("Solution found:")
            for state in solution:
                print(state)
        else:
            print("No solution found or solution is invalid.")

    # Generate and use random initial states
    for i in range(domain.GAME_CONFIG['NUM_RANDOM_STATES']):
        initial_state = domain.generate_random_initial_state()
        print(f"\nSolving for random initial state {i+1}: {initial_state}")
        solution = bfs_search(domain, succ_code, goal_code, initial_state)
        if domain.validate_solution(solution):
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
    print(f"Initial states: {domain.GAME_CONFIG['INITIAL_STATES']}")
    print(f"Number of random states: {domain.GAME_CONFIG['NUM_RANDOM_STATES']}")
    print(f"Max iterations for refining functions: {CONFIG['MAX_ITERATIONS']}")
    print(f"Timeout for function execution: {CONFIG['FUNCTION_TIMEOUT']} second")

if __name__ == '__main__':
    main()
