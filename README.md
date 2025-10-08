# End-to-End Verification Project for a CNN Accelerator Core

![Demo](images/practice.gif)

This project simulates the verification process of an AI "chip" (Device Under Test - DUT), written in C++, by comparing its results against a "Golden Model" written in Python.

Minst Training and the model is define in the Link (example computer vision model for testing): [Google Colab](https://colab.research.google.com/drive/1KsgzXknRXHkg0Nan6ViOg_8l6MMmQy7X)

## Structure

- `dut_cpp/`: C++ code that simulates the hardware constraints of a quantized CNN.
- `pybind_wrapper/`: C++ code using Pybind11 to create a bridge between C++ and Python.
- `python_testbench/`: The main verification environment.
  - `golden_model.py`: The reference model written in NumPy, considered the "correct answer".
  - `run_test.py`: The main script that handles training, quantization, running both models, and comparing the results.
- `CMakeLists.txt`: The build file for the C++ components.

## How Does the Simulated AI Chip (DUT) Work?

Imagine that instead of building an expensive physical chip, I write a C++ program that "mimics" exactly how that chip would behave. This is like an architect creating a 3D model of a house on a computer before actually building it. It allows to find design issues earlier and more cheaply.

My simulated chip is designed to be extremely fast and power-efficient, so it has two key characteristics:

#### 1. Quantization - "Compressing" Information

AI models on computers often use very detailed numbers (floating-point numbers, e.g., 0.1234567). But to make a chip run fast and stay cool, it only uses simple whole numbers within a very small range, for example, from **-128 to 127** (called an `int8_t`).

This process is like repainting a photo with millions of colors using only a 256-crayon box. You have to pick the closest available color. This "compresses" the information, helping the chip process data much faster.

```
          |-------------------------------------| (Floating-point numbers, very precise)
         -1.0                                +1.0

                       ▼ QUAN-TI-ZA-TION ▼

| ... -3, -2, -1, 0, 1, 2, 3 ... | (8-bit integers, simple)
-128                             127
```

#### 2. The Multiply-Accumulate (MAC) Unit - The "Heart" of the Chip

The most common operation in AI is multiplying a series of numbers and then adding them all together. Our C++ "chip" simulates a specialized hardware block to do exactly this.

- **Inputs (`Input` and `Weight`):** These are `int8_t` integers (the result of quantization).
- **Multiplication (`*`):** When two `int8_t` numbers are multiplied, the result can be larger, so I need a bigger "box" to hold it, which is an `int16_t`.
- **Addition (`+`):** When I add hundreds of these multiplication results together, the total sum can become very large. I need a huge "bucket" to ensure the number doesn't overflow, which is an `int32_t`.

A diagram of this simulated architecture is as follows:

```
Input                 Weight
 (int8_t)              (int8_t)
    │                     │
    └─────────┬───────────┘
              │
             (*) --Multiplication--> Product
                                       (int16_t)
                                          │
              ┌─────────────────────────┘
              │
Accumulator <-(+) -----Addition----- Previous Accumulation
  (int32_t)   │                         (int32_t)
```
Simulating these data sizes (`int8_t`, `int16_t`, `int32_t`) precisely is critical to ensure our C++ program behaves exactly like a real chip.

## Why Do I Need Both a DUT and a Golden Model?

To verification

- **Golden Model (Python):** I Think of this as the "Master Chef". This chef uses the best tools available (a powerful computer, super-accurate `float32` numbers) to create a "perfect answer". The Golden Model's result is always considered 100% correct.
- **DUT - Simulated Chip (C++):** This is the "Apprentice Chef". This apprentice must use a simpler, more efficient toolkit ( `int8_t` numbers, hardware constraints) to produce a result.

**my job** is to compare the apprentice's result with the master's. If the result is "close enough" to the perfect answer, it means our chip design is correct!

```
                                 ┌──────────────────────────┐
      Input Image                │   Golden Model (Python)  │
      ───────────────>           │ (The Expert, High Precision) │──────┐
           │                     └──────────────────────────┘      │
           │                                                       │
           │                     ┌──────────────────────────┐      ▼         ┌───────────┐
           └───────────────────> │      DUT - C++ Chip      │───> Compare   │ PASS/FAIL │
                                 │ (The Apprentice, Speed-Optimized) │       └───────────┘
                                 └──────────────────────────┘      ▲
                                                                   │
                                           Is the result close enough?
```

## Requirements

- C++ compiler (g++, clang, MSVC)
- CMake (>= 3.12)
- Python (>= 3.8)
- pip

## Installation and Execution Guide

1.  **Clone the repository:**
    ```bash
    git clone 
    cd cnn_verification_project
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Compile the C++ module:**
    ```bash
    # Install build tools (one-time setup on Ubuntu/Debian)
    # sudo apt install build-essential cmake python3-dev pybind11-dev

    # Create a build directory
    mkdir -p build
    cd build

    # Run CMake to prepare and then compile
    cmake ..
    make -j$(nproc) # Use all CPU cores for a faster compilation

    cd ..
    ```
    This command will create a `cnn_dut_module.so` file (or `.pyd` on Windows) inside the `build/` directory.

4.  **Copy the file mnist_cnn.h5 to python_testbench folder:**
    access to [Model_parameter](https://drive.google.com/file/d/1OuxesB2e50_5x-yAT4WKmyGOvrypqb4k/view?usp=sharing).

    The first time you run this, it will take a few minutes to train the model (if you dont' requirement mnist_cnn.h5). Subsequent runs will automatically load the saved model.


5.  **Run the verification script:**
    ```bash
    python python_testbench/run_test.py
    ```
    The first time you run this, it will take a few minutes to train the model. Subsequent runs will automatically load the saved model.

6.  **Run the streamlit script:**
    ```bash
    python streamlit run python_testbench/visual_tester.py
    ```
## Analyzing the Results

The script will run a test on a random image and report one of the following:
- **PASS**: The DUT's prediction matches the Golden Model's, and the output probability error is negligible. This means our chip design is working perfectly.
- **PARTIAL PASS**: The predictions match, but there is a noticeable difference in the output probabilities. This indicates that hardware constraints (like quantization) are causing some precision loss, but not enough to change the final answer.
- **FAIL**: The predictions do not match. This is a critical issue, suggesting that our hardware design may be causing too much error, leading to an incorrect result.

The job of a verification engineer is to analyze the differences between the `Golden Model Output` and the `DUT C++ Output` to fully understand the impact of hardware design decisions.
