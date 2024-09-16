# AutoToS-Inspired: Automating Thought of Search

This project implements an approach inspired by AutoToS (Automating Thought of Search), a general method for automatically generating sound and complete search components for planning and reasoning problems using large language models (LLMs).

### THIS WAS BASICALLY ENTIRELY AI-GENERATED

Credits to GPT-o1-preview for initial code extraction from source paper and implementation, and Cursor (www.cursor.com) + Claude-3.5-sonnet for subsequent iteration and error testing.  Basically all suggested code was used, basically blindly, but the monkey man did a little bit of direction.

### NEW SUBSEQUENT IMPLEMENTATIONS SOON.  FRUSTRATING MONKEY TIME BOTTLENECK

But this does work.  Try it with gpt-4o-mini to see it fail due to insufficient intelligence.  gpt-3.5-turbo at least has a chance of failure.  Will test with local weaker models soon to see how low it can go, but more challenging implementation problems first.

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