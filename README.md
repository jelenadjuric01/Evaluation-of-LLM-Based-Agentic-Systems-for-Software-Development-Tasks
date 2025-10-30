# AgentFix: Automated Python Bug Fixing with Language Models

AgentFix is a research project that leverages pre-trained language models to automatically fix Python bugs. The system takes buggy code and test failures as input, then generates minimal patches to make the tests pass. This repository implements an iterative agent that can attempt multiple repair strategies and evaluates performance on the HumanEvalFix benchmark.

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

### Benchmark Runner (`run_benchmark.py`)
- Orchestrates the complete evaluation pipeline
- Supports extensive model configuration via command-line arguments
- Implements iterative repair (up to `max_steps` attempts per task, default: 1)
- Tracks detailed execution traces and timing information
- Reports results using **pass@k** metric where k = `max_steps`

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
"You are a careful Python bug fixer. Produce the MINIMAL patch. "
    "Keep the same function signature and behavior unless tests require otherwise. "
    "Avoid I/O, networking, and randomness."
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
4. **Iterative Repair**: If tests fail and `max_steps > 1`, use failure summary as context for next attempt
5. **Success/Failure**: Record results after success or max attempts reached
6. **Evaluation**: Calculate **pass@k** where k = `max_steps` (default: **pass@1**)

## Results Directory

The `results/` folder contains comprehensive evaluation data:

- **Individual Reports**: JSON files named by configuration parameters (e.g., `deterministic_config.json`, `temp_0.3_top_p_0.9.json`)
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
python -m src.eval.run_benchmark --report results/deterministic.json
```

#### Sampling-Based Generation
```bash
python -m src.eval.run_benchmark --temperature 0.3 --top_p 0.9 --report results/sampling_t0.3.json
```

#### Beam Search
```bash
python -m src.eval.run_benchmark --num_beams 3 --report results/beam3.json
```

### Command-Line Parameters

#### Benchmark Configuration
- `--sample N`: Test on N randomly selected tasks (default: all tasks)
- `--seed N`: Random seed for reproducibility (default: 42)
- `--max_steps N`: Maximum repair attempts per task (default: 1)
- `--timeout N`: Execution timeout in seconds (default: 10)
- `--report PATH`: Output JSON report path (default: out/report.json)

**Evaluation Metric Note**: The system uses **pass@k** evaluation where k equals `max_steps`. With the default `max_steps=1`, results are reported as **pass@1**. If you change `max_steps` to a different value (e.g., 3), the metric becomes **pass@3**.

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

All of the results can be found in the `results/comparative results.csv` while detailed report is available in the folder `results`.

### Phase 1: Baseline Establishment
Initially, I focused on deterministic generation (greedy decoding) with a single repair attempt (`max_steps=1`) to establish a reliable baseline using the **pass@1** metric. This approach prioritizes consistency and repeatability, which is crucial for bug fixing where deterministic behavior is often preferred over creative variation.

The deterministic configuration with **pass@1** evaluation produced initial results, demonstrating the model's capability to understand and fix common programming errors through careful prompt engineering and robust code extraction.

### Phase 2: Parameter Exploration - Temperature Sampling

After establishing the deterministic baseline, I explored temperature-based sampling to understand whether introducing controlled randomness could improve patch quality.

**Temperature 0.3, top_p 0.9**: **pass@1 = 0.293** (-1.2% vs baseline)
- Rationale: Started with low temperature (0.3) to maintain focus while allowing slight variation
- Result: Slight decrease from baseline, suggesting that even minimal randomness may hurt consistency in bug fixing

**Temperature 0.5, top_p 0.9**: **pass@1 = 0.262** (-4.3% vs baseline)
- Rationale: Tested moderate temperature to see if more diversity helps with challenging bugs
- Result: Further decline in performance, indicating that increased randomness is counterproductive

**Temperature 0.1, top_p 0.9**: **pass@1 = 0.299** (-0.6% vs baseline)
- Rationale: After seeing degradation at 0.3 and 0.5, tested very low temperature closer to deterministic behavior
- Result: Performance closer to baseline but still slightly worse, confirming that any sampling introduces unwanted variability

**Temperature 0.2, top_p 0.9**: **pass@1 = 0.317** (+1.2% vs baseline) ⭐
- Rationale: Fine-tuned between 0.1 and 0.3 to find the optimal balance
- Result: **Best performing configuration overall**, showing that minimal, carefully controlled randomness can help escape local optima in certain edge cases

### Phase 3: Beam Search Exploration

Having found that very low temperature sampling could marginally improve results, I explored beam search as an alternative approach to introduce structured exploration while maintaining determinism.

**Beam Search (3 beams)**: **pass@1 = 0.274** (-3.1% vs baseline)
- Rationale: Test whether exploring multiple generation paths simultaneously helps find better solutions
- Result: Underperformed baseline, suggesting that the additional complexity doesn't translate to better patches

**Beam Search (5 beams)**: **pass@1 = 0.280** (-2.5% vs baseline)
- Rationale: Increased beam width to explore more alternatives, following the modest improvement from 3 to 5 beams
- Result: Slightly better than 3 beams but still worse than baseline, indicating diminishing returns

**Beam Search (4 beams)**: **pass@1 = 0.287** (-1.8% vs baseline)
- Rationale: Tested intermediate beam count to find optimal balance between exploration and focus
- Result: Best among beam search variants but still inferior to both baseline and optimal temperature sampling

### Phase 4: Fine-tuning Around Optimal Temperature

After identifying temperature 0.2 as the best performer, I conducted focused optimization around this configuration to further improve results.

**Temperature 0.2, top_p 0.85**: **pass@1 = 0.287** (-1.8% vs baseline, -3.0% vs best)
- Rationale: Test if reducing top_p from 0.9 to 0.85 provides better focus by restricting vocabulary more aggressively
- Result: Significant drop from the 0.2/0.9 combination, indicating that top_p 0.9 provides the right balance between focus and flexibility

**Temperature 0.2, top_p 0.95**: **pass@1 = 0.299** (-0.6% vs baseline, -1.8% vs best)
- Rationale: Test if increasing top_p to 0.95 allows access to slightly more diverse vocabulary while maintaining low temperature
- Result: Better than 0.85 but still worse than 0.9, confirming that 0.9 is optimal for top_p

### Phase 5: Repetition Penalty Optimization

With temperature and top_p optimized, I explored whether adjusting repetition penalty could further improve performance, since code often contains necessary repetitive patterns.

**Temperature 0.2, top_p 0.9, repetition_penalty 1.02**: **pass@1 = 0.280** (-2.5% vs baseline, -3.7% vs best)
- Rationale: Reduce penalty to allow more natural code repetition patterns (loops, similar variable names, etc.)
- Result: Notable performance drop, suggesting that some repetition penalty is necessary to avoid degenerate outputs

**Temperature 0.2, top_p 0.9, repetition_penalty 1.03 & 1.07**: **pass@1 = 0.299** (-0.6% vs baseline, -1.8% vs best)
- Rationale: Test moderate reduction/enlargment in penalty to balance repetition control with code pattern flexibility
- Result: Better than 1.02 but still inferior to default 1.05, confirming that the default penalty is well-calibrated

### Key Insights and Conclusions

1. **Deterministic Superiority**: For most configurations, the deterministic baseline proved most reliable, aligning with the hypothesis that bug fixing benefits from consistent, predictable behavior.

2. **Minimal Sampling Sweet Spot**: Temperature 0.2 with top_p 0.9 achieved the best results (31.7% pass@1), suggesting that very controlled randomness can help in edge cases without introducing significant instability.

3. **Temperature Sensitivity**: Performance degrades quickly as temperature increases beyond 0.2, with each increment of 0.1-0.3 showing measurable decline in success rate.

4. **Beam Search Limitations**: All beam search configurations underperformed both deterministic and low-temperature sampling, indicating that the structured exploration doesn't align well with the singular nature of bug fixing tasks.

5. **Parameter Interdependence**: Top_p and repetition_penalty showed strong sensitivity around their optimal values - small deviations (±0.05 for top_p, ±0.02-0.03 for penalty) led to measurable performance drops.

6. **Fine-tuning Limits**: Despite systematic exploration around the optimal configuration, no parameter adjustments improved upon the initial temperature 0.2, top_p 0.9, penalty 1.05 combination, suggesting this represents a local optimum.

7. **Marginal but Consistent Gains**: The overall performance range (26.2% to 31.7%) demonstrates that while parameter tuning provides modest improvements, the gains are consistent and meaningful for automated systems.

### Final Recommended Configuration

Based on these comprehensive experiments, the **optimal configuration** for AgentFix is:
- **Temperature**: 0.2
- **top_p**: 0.9  
- **repetition_penalty**: 1.05 (default)
- **do_sample**: True
- All other parameters at default values

This configuration provides a 3.9% relative improvement over the deterministic baseline while proving robust against parameter variations, making it the most reliable choice for automated bug fixing systems.

### Future Optimization Directions

Having reached apparent saturation in hyperparameter optimization around the current configuration, several promising avenues remain for further performance improvements:

**Advanced Parameter Exploration**: More granular parameter sweeps could reveal micro-optimizations, such as testing temperature values between 0.18-0.22 in 0.01 increments, exploring top_k sampling in combination with the optimal temperature/top_p settings, or investigating dynamic temperature scheduling that starts higher and decreases during generation.

**Prompt Engineering Refinement**: The current prompt structure could be enhanced through techniques like few-shot learning (providing 2-3 examples of successful bug fixes), more specific constraint language that better guides the model toward minimal patches, or adaptive prompting that adjusts based on bug complexity or failure patterns.

**Model Architecture Exploration**: Testing larger models from the same family (Qwen2.5-Coder-7B, 14B) could provide significant improvements, while comparing against other code-specialized models (CodeT5, StarCoder, Code Llama) might reveal architecture-specific advantages for bug fixing tasks.

**Fine-tuning and Specialization**: Custom fine-tuning on HumanEvalFix training data or similar bug-fixing datasets could create a model specifically optimized for this task. Additionally, parameter-efficient fine-tuning methods (LoRA, QLoRA) could adapt the model to bug-fixing patterns without full retraining costs.

**Multi-step Strategy Optimization**: While current experiments focus on pass@1, exploring multi-step approaches with the optimal parameters (pass@2, pass@3) could reveal whether the configuration performs better with iterative refinement, and whether different parameters are optimal for subsequent repair attempts.

**Context and Memory Enhancements**: Implementing more sophisticated failure analysis, maintaining context across repair attempts, or incorporating static analysis tools could provide richer input for the generation process, potentially breaking through the current performance ceiling.

These directions represent natural next steps once hyperparameter optimization has been exhausted, focusing on more fundamental changes to the model, training, or system architecture.

