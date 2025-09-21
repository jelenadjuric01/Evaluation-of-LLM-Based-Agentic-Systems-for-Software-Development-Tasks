# AgentFix: Automated Python Bug Fixing with Language Models

This is a research project that leverages pre-trained language models to automatically fix Python bugs. The system takes buggy code and test failures as input, then generates minimal patches to make the tests pass. This repository implements an iterative agent that can attempt multiple repair strategies and evaluates performance on the HumanEvalFix benchmark.

## Project Structure

```

├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   └── policy.py              # Core LLM-based bug fixing logic
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── humanevalfix.py        # HumanEvalFix dataset loader
│   │   ├── run_benchmark.py       # Main evaluation script
│   │   └── run_one.py            # Single task testing utility
│   └── sandbox/
│       ├── __init__.py
│       └── runner.py             # Safe code execution environment
├── results/
│   ├── *.json                    # Individual experiment reports
│   └── comparative_results.csv   # Aggregated results across configurations
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## Core Components

### 1. Agent Policy (`src/agent/policy.py`)

The core bug-fixing logic uses a pre-trained language model to generate patches. Key features:

- **Model Configuration**: Flexible configuration system supporting different generation strategies (greedy, sampling, beam search)
- **Prompt Engineering**: Carefully crafted system and user prompts that emphasize minimal patches
- **Code Extraction**: Robust parsing to extract clean Python definitions from LLM output
- **Fallback Safety**: Returns original buggy code if extraction fails, preventing import errors

#### Model Selection: Qwen2.5-Coder-1.5B-Instruct

I chose **Qwen/Qwen2.5-Coder-1.5B-Instruct** as our default model for several reasons:

1. **Code Specialization**: Specifically fine-tuned for code generation and understanding
2. **Size Efficiency**: 1.5B parameters provide good performance while being computationally manageable
3. **Instruction Following**: The "Instruct" variant is optimized for following detailed prompts
4. **CPU Compatibility**: Can run on CPU when CUDA isn't available, making it accessible
5. **Recent Training**: Part of the Qwen2.5 series with up-to-date coding knowledge

### 2. Evaluation Framework (`src/eval/`)

#### HumanEvalFix Dataset (`humanevalfix.py`)
- Loads the BigCode HumanEvalPack Python test split
- Contains buggy functions with corresponding test suites
- Each task includes: buggy code, tests, and function entry point

#### Benchmark Runner (`run_benchmark.py`)
- Orchestrates the complete evaluation pipeline
- Supports extensive model configuration via command-line arguments
- Implements iterative repair (up to `max_steps` attempts per task)
- Tracks detailed execution traces and timing information

#### Single Task Tester (`run_one.py`)
- Utility for testing individual tasks during development
- Includes a sample palindrome bug for quick validation

### 3. Sandbox Execution (`src/sandbox/runner.py`)

Safe execution environment that:
- Creates isolated temporary directories for each test
- Writes candidate code and tests to separate files
- Executes tests directly (no pytest dependency)
- Captures stdout/stderr and enforces timeouts
- Cleans up resources automatically

## Prompt Engineering Strategy

My prompt design follows several key principles:

### System Prompt
```
"You are a careful Python bug fixer. Produce the MINIMAL patch. 
Keep the same function signature and behavior unless tests require otherwise. 
Avoid I/O, networking, and randomness."
```

### User Prompt Structure
1. **Clear Goal**: "Produce a MINIMAL patch that makes the tests pass"
2. **Strict Constraints**: 
   - Exact signature preservation
   - No helper components or additional complexity
   - No I/O or non-deterministic behavior
3. **Format Requirements**: Raw Python source only, no markdown
4. **Context Provision**: Buggy code + failure summary

This approach encourages focused, surgical fixes rather than complete rewrites.

## Generation Parameters & Strategies

The system supports three main generation strategies:

### 1. Greedy/Deterministic (Default)
- `do_sample=False`
- `num_beams=1`
- Most reliable for consistent, focused patches
- Used in initial experiments

### 2. Sampling-Based
- `do_sample=True`
- Configurable `temperature`, `top_p`, `top_k`
- Provides diversity at the cost of potential inconsistency

### 3. Beam Search
- `num_beams > 1`
- `early_stopping=True`
- Balances exploration with deterministic selection

#### Parameter Rationale
- **`max_new_tokens=1024`**: Sufficient for most patches without excessive generation
- **`repetition_penalty=1.05`**: Mild penalty to avoid loops without suppressing necessary repetition
- **No n-gram blocking**: Python code naturally contains repeated patterns

## Workflow

1. **Task Loading**: Load buggy code and tests from HumanEvalFix
2. **Initial Generation**: Generate patch using LLM with no failure context
3. **Execution**: Run generated code against test suite in sandbox
4. **Iterative Repair**: If tests fail, use failure summary as context for next attempt
5. **Success/Failure**: Record results after success or max attempts reached

## Results Directory

The `results/` folder contains comprehensive evaluation data:

- **Individual Reports**: JSON files named by configuration parameters (e.g., `deterministic_config.json`, `temp_0.7_top_p_0.9.json`)
- **Comparative Analysis**: `comparative_results.csv` aggregates key metrics across all experiments
- **Full Traces**: Each report includes complete execution traces with intermediate attempts

## Getting Started

### Installation

#### Requirements
- Python 3.8 or higher
- PyTorch (automatically installs with transformers)
- Hugging Face Transformers
- Datasets library

#### Cross-Platform Setup

**Windows (Command Prompt/PowerShell):**
```cmd
git clone https://github.com/jelenadjuric01/Evaluation-of-LLM-Based-Agentic-Systems-for-Software-Development-Tasks.git
cd Evaluation-of-LLM-Based-Agentic-Systems-for-Software-Development-Tasks
pip install -r requirements.txt
```

**macOS/Linux (Terminal):**
```bash
git clone https://github.com/jelenadjuric01/Evaluation-of-LLM-Based-Agentic-Systems-for-Software-Development-Tasks.git
cd Evaluation-of-LLM-Based-Agentic-Systems-for-Software-Development-Tasks
pip install -r requirements.txt
```

**Google Colab/Kaggle (GPU-enabled):**
```python
# Clone and setup
!git clone https://github.com/jelenadjuric01/Evaluation-of-LLM-Based-Agentic-Systems-for-Software-Development-Tasks.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -r requirements.txt

# Run experiments with GPU acceleration
!python -m src.eval.run_benchmark --temperature 0.4 --top_p 0.9 --report out/val_temp_0.4.json
```

### Quick Test
```bash
python -m src.eval.run_one
```

### Full Evaluation Examples

#### Deterministic Generation (Default)
```bash
python -m src.eval.run_benchmark --sample 10 --report results/deterministic.json
```

#### Sampling-Based Generation
```bash
python -m src.eval.run_benchmark --sample 10 --temperature 0.7 --top_p 0.9 --report results/sampling_t0.7.json
```

#### Beam Search
```bash
python -m src.eval.run_benchmark --sample 10 --num_beams 3 --report results/beam3.json
```

### Command-Line Parameters

#### Benchmark Configuration
- `--sample N`: Test on N randomly selected tasks (default: all tasks)
- `--seed N`: Random seed for reproducibility (default: 42)
- `--max_steps N`: Maximum repair attempts per task (default: 4)
- `--timeout N`: Execution timeout in seconds (default: 10)
- `--report PATH`: Output JSON report path (default: out/report.json)

#### Model Configuration
- `--model_id MODEL`: Hugging Face model identifier (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)
- `--device DEVICE`: Computation device - cuda/cpu (default: auto-detect)
- `--max_new_tokens N`: Maximum tokens to generate per attempt (default: 1024)
- `--repetition_penalty FLOAT`: Penalty for repetitive text (default: 1.05)

#### Generation Strategy (Mutually Exclusive)
- `--temperature FLOAT`: Enable sampling with specified temperature (0.1-2.0)
- `--num_beams N`: Enable beam search with N beams

#### Sampling Parameters (Only with --temperature)
- `--top_p FLOAT`: Nucleus sampling threshold (default: 0.9)
- `--top_k N`: Top-k sampling limit (optional)

### Usage Examples by Operating System

**Windows PowerShell:**
```powershell
# Full evaluation with sampling
python -m src.eval.run_benchmark --sample 50 --temperature 0.5 --top_p 0.95 --report "results\temp_0.5.json"

# Beam search experiment
python -m src.eval.run_benchmark --sample 25 --num_beams 5 --report "results\beam5.json"
```

**macOS/Linux Bash:**
```bash
# Batch experiment script
for temp in 0.3 0.5 0.7; do
    python -m src.eval.run_benchmark \
        --sample 30 \
        --temperature $temp \
        --report results/temp_${temp}.json
done
```

**Kaggle Notebook (GPU):**
```python
# Systematic parameter exploration
temperatures = [0.3, 0.4, 0.5, 0.6, 0.7]

for temp in temperatures:
    !python -m src.eval.run_benchmark \
        --sample 100 \
        --temperature {temp} \
        --top_p 0.9 \
        --report out/kaggle_temp_{temp}.json
    print(f"Completed temperature {temp}")
```

## Experimental Progression

### Phase 1: Baseline Establishment
Initially, we focused on deterministic generation (greedy decoding) to establish a reliable baseline. This approach prioritizes consistency and repeatability, which is crucial for bug fixing where deterministic behavior is often preferred over creative variation.

The deterministic configuration produced promising initial results, demonstrating the model's capability to understand and fix common programming errors through careful prompt engineering and robust code extraction.

### Phase 2: Parameter Exploration *(In Progress)*
After validating the core approach with deterministic generation, we began systematic exploration of generation parameters to understand their impact on fixing accuracy and patch quality. This includes:

- Temperature scaling for controlled randomness
- Top-p and top-k sampling for vocabulary restriction
- Beam search for structured exploration
- Various combinations and their trade-offs

*[Detailed results and analysis of parameter tuning experiments will be documented here as they become available]*

## Key Design Decisions

1. **Minimal Patches**: Emphasize surgical fixes over rewrites to maintain code maintainability
2. **Single Component Scope**: Focus on individual code units rather than multi-file changes
3. **No External Dependencies**: Self-contained execution without additional libraries
4. **Robust Extraction**: Multiple fallback strategies for parsing LLM output
5. **Comprehensive Logging**: Detailed traces for analysis and debugging

## Future Work

- Integration with additional code models (CodeT5, StarCoder, etc.)
- Multi-turn conversation for complex debugging
- Integration with static analysis tools
- Evaluation on additional benchmarks (Defects4J, etc.)
- Real-world deployment studies

---

**Note**: This is a research prototype. Generated patches should be carefully reviewed before production use.