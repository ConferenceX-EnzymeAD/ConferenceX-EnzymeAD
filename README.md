# Performance of EnzymeAD in Real World High-Complexity Deep Learning Models

This repository contains the final submission for CSC2516, including experiments and implementations related to training GPT-2 and layer-wise optimization.

## Repository Structure

- **`train_gpt2.c`**: Contains the main C implementation for training GPT-2.
- **`train_gpt2_layer.c`**: Focuses on layer-wise training of GPT-2.
- **`train_gpt2_orig.c`**: Includes the original GPT-2 training implementation.
- **`src/main.rs`**: The Rust implementation leveraging the Enzyme autodiff tool for differentiable programming.
- Other support files and build configuration files for setting up and running the experiments.

## Prerequisites and Setup

1. **Install Required Tools**:  
   The Rust compiler, C compiler, and Enzyme autodiff tool needed for this experiment can be installed by following the instructions [here](https://enzyme.mit.edu/rust/installation.html).

2. **Install Google-Benchmark**:  
   Follow the setup instructions provided by Google-Benchmark.

3. **Enable OpenMP Support (Optional)**:  
   Experiments with OpenMP support require modifying the `CMakeCache.txt` file to enable the OpenMP runtime and recompiling the project.

## Key Code Files

The primary code of interest is located in the following files:

- **`llm.c/train_gpt2.c`**
- **`llm.c/train_gpt2_layer.c`**
- **`llm.c/train_gpt2_orig.c`**
- **`llmrs/src/main.rs`**

## Running the Experiments

After setting up the required tools and dependencies, you can run the experiments by compiling and executing the appropriate source files.

Refer to the comments in each file for specific instructions on usage and experiment details.

---

For additional details or issues, feel free to open an issue or contact the repository contributors.
