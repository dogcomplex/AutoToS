# AutoToS-Inspired: Automating Thought of Search

This project implements an approach inspired by AutoToS (Automating Thought of Search), a general method for automatically generating sound and complete search components for planning and reasoning problems using large language models (LLMs).

### THIS WAS BASICALLY ENTIRELY AI-GENERATED

Credits to GPT-o1-preview for initial code extraction from source paper and implementation, and Cursor (www.cursor.com) + Claude-3.5-sonnet for subsequent iteration and error testing.  Basically all suggested code was used, basically blindly, but the monkey man did a little bit of direction.

### NEW SUBSEQUENT IMPLEMENTATIONS SOON.  FRUSTRATING MONKEY TIME BOTTLENECK

EDIT: lol nevermind still a bit overconfident.   Does not pass all unit tests yet - might still be mix of printout quality or actual verification correctness.  Need to iterate a bit more, and maybe use a genie wish (GPT-o1) to fix.  

## Overview

This implementation is inspired by the concepts presented in "Automating Thought of Search: A Journey Towards Soundness and Completeness" by Cao et al. (arXiv:2408.11326). While the high-level approach is based on their work, this implementation represents our own original code and design decisions.

Our approach builds upon the Thought of Search (ToS) framework, automating the process of generating correct code for successor functions and goal tests without human-in-the-loop feedback. It uses unit tests, partial soundness checks, and iterative refinement to guide LLMs in producing high-quality search components.

Currently implemented domains:
- 24 Game (fully implemented example)
- BlocksWorld
- Mini Crosswords
- PrOntoQA
- Sokoban

## Key Features

- Automated feedback loop for refining search components
- Support for multiple LLM backends (GPT-4, LLaMA, DeepSeek)
- Domain-independent and domain-specific unit tests
- Partial soundness checks for early error detection
- BFS/DFS search algorithms with generated components

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key in the `CONFIG` dictionary in `autotos.py`

## Usage

Run the main script:
```
python autotos.py
```

This will generate the necessary search components for the 24 Game (or other implemented domains) and attempt to find solutions for the provided problem instances.

## Configuration

You can modify the following parameters in the `CONFIG` dictionary:

- `OPENAI_API_KEY`: Your OpenAI API key
- `MODEL_NAME`: The LLM to use (e.g., 'gpt-4o-mini', 'gpt-4o')
- `MAX_ITERATIONS`: Maximum number of iterations for refining functions
- `FUNCTION_TIMEOUT`: Timeout for function execution in seconds
- `INITIAL_STATE`: The initial state for the problem (domain-specific)

## Extending AutoToS-Inspired

To add support for new domains:

1. Implement domain-specific unit tests
2. Define a partial soundness check function
3. Provide initial prompts for successor and goal functions
4. Implement a solution validation method

### Example Output

```
Refining goal function...
LLM query took 2.87 seconds
Received code:
def isgoal(state):
    def evaluate(expression):
        try:
            return eval(expression) == 24
        except ZeroDivisionError:
            return False

    def isgoal_helper(numbers):
        if len(numbers) == 1:
            return numbers[0] == 24
        for i in range(len(numbers)):
            for j in range(len(numbers)):
                if i != j:
                    new_numbers = [numbers[k] for k in range(len(numbers)) if k != i and k != j]
                    if isgoal_helper(new_numbers + [f'({numbers[i]}+{numbers[j]})']) or \
                       isgoal_helper(new_numbers + [f'({numbers[i]}-{numbers[j]})']) or \
                       isgoal_helper(new_numbers + [f'({numbers[i]}*{numbers[j]})']) or \
                       (numbers[j] != 0 and isgoal_helper(new_numbers + [f'({numbers[i]}/{numbers[j]})']):
                        return True
        return False

    return isgoal_helper(state)
Goal function failed: Exception occurred when defining isgoal: invalid syntax (<string>, line 18)
LLM query took 2.64 seconds
Received code:
def isgoal(state):
    def evaluate(expression):
        try:
            return eval(expression) == 24
        except ZeroDivisionError:
            return False

    def isgoal_helper(numbers):
        if len(numbers) == 1:
            return numbers[0] == 24
        for i in range(len(numbers)):
            for j in range(len(numbers)):
                if i != j:
                    new_numbers = [numbers[k] for k in range(len(numbers)) if k != i and k != j]
                    if isgoal_helper(new_numbers + [f'({numbers[i]}+{numbers[j]})']) or \
                       isgoal_helper(new_numbers + [f'({numbers[i]}-{numbers[j]})']) or \
                       isgoal_helper(new_numbers + [f'({numbers[i]}*{numbers[j]})']) or \
                       (numbers[j] != 0 and isgoal_helper(new_numbers + [f'({numbers[i]}/{numbers[j]})'])):
                        return True
        return False

    return isgoal_helper(state)
Goal function passed all tests.
Refining successor function...
LLM query took 4.42 seconds
Received successor code:
from itertools import combinations

def succ(state):
    def apply_operation(num1, num2, op):
        if op == '+':
            return num1 + num2
        elif op == '-':
            return num1 - num2
        elif op == '*':
            return num1 * num2
        elif op == '/' and num2 != 0:
            return num1 / num2
        else:
            return None

    successor_states = []
    for num1, num2 in combinations(state, 2):
        for op in ['+', '-', '*', '/']:
            result = apply_operation(num1, num2, op)
            if result is not None and result not in state and result not in successor_states:
                new_state = state.copy()
                new_state.remove(num1)
                new_state.remove(num2)
                new_state.append(result)
                successor_states.append(new_state)

    return successor_states
Successor function failed: Exception in succ function on input [6, 6, 6, 6]: name 'combinations' is not defined
LLM query took 2.76 seconds
Received successor code:
def succ(state):
    def apply_operation(num1, num2, op):
        if op == '+':
            return num1 + num2
        elif op == '-':
            return num1 - num2
        elif op == '*':
            return num1 * num2
        elif op == '/' and num2 != 0:
            return num1 / num2
        else:
            return None

    from itertools import combinations  # Import moved inside the function

    successor_states = []
    for num1, num2 in combinations(state, 2):
        for op in ['+', '-', '*', '/']:
            result = apply_operation(num1, num2, op)
            if result is not None and result not in state and result not in successor_states:
                new_state = state.copy()
                new_state.remove(num1)
                new_state.remove(num2)
                new_state.append(result)
                successor_states.append(new_state)

    return successor_states
Successor function passed all tests.
Starting BFS search...
Solution found:
[6, 6, 6, 6]
[6, 6, 12]
[6, 18]
[24]

Summary Statistics:
Total execution time: 12.75 seconds
LLM model used: gpt-3.5-turbo
Total LLM queries: 4
Total tokens used: 2098

Function call counts:
  get_code_from_api: 4
  test_goal_function: 2
  test_successor_function: 2
  bfs_search: 1

Other important parameters:
Initial state: [6, 6, 6, 6]
Max iterations for refining functions: 100
Timeout for function execution: 1 second
```



## Citation

If you use this code in your research, please cite both this repository and the original paper that inspired it:

```
@misc{cao2024automating,
title={Automating Thought of Search: A Journey Towards Soundness and Completeness},
author={Daniel Cao and Michael Katz and Harsha Kokel and Kavitha Srinivas and Shirin Sohrabi},
year={2024},
eprint={2408.11326},
archivePrefix={arXiv},
primaryClass={cs.AI},
url={https://arxiv.org/abs/2408.11326}
}
```

## License

This project is open-source and available under the MIT License. See the LICENSE file for more details.
