# Hybrid-Calo-UDE

**Hybrid-Calo-UDE** is a Julia (Lux) implementation of a hybrid Graph Neural Network (GNN) and 3D Convolutional Neural Network (CNN) model designed to generate smooth and robust ensemble mean profiles of Electromagnetic Calorimeter (ECAL) showers.

This specific implementation is the **v32 STABLE MEAN** configuration, which is tuned to guarantee convergence to smooth profiles by adjusting weights to prevent common issues like "ReLU death" and "black heatmaps."

---

## 🚀 Getting Started

This project is built as a standard Julia package for complete reproducibility.

### Prerequisites
* [**Julia (v1.6 or later)**](https://julialang.org/downloads/)
* [**Git**](https://git-scm.com/downloads)
* The **HDF5 dataset** (e.g., `dataset_2_1.hdf5`) must be manually placed in the `/data` folder.

### 1. Installation
First, clone the repository to your local machine:
```bash
git clone [https://github.com/YourUsername/Hybrid-Calo-UDE.git](https://github.com/YourUsername/Hybrid-Calo-UDE.git)
cd Hybrid-Calo-UDE
````

### 2\. Environment Setup

Next, install all the required Julia packages using the project's built-in environment.

1.  Start the Julia REPL:
    ```bash
    julia
    ```
2.  Press `]` to enter the package manager.
3.  Activate the project environment:
    ```pkg
    pkg> activate .
    ```
4.  Instantiate the environment. This will read the `Manifest.toml` file and install the *exact* versions of all dependencies used for this project, ensuring perfect reproducibility.
    ```pkg
    pkg> instantiate
    ```
5.  Press `Backspace` to exit the package manager.

-----

## 🏃‍♂️ How to Run

After installing the dataset and setting up the environment, you can run the entire training and evaluation pipeline with a single command.

Execute the main training script from your terminal (make sure you are in the `Hybrid-Calo-UDE` directory):

```bash
julia --project scripts/train.jl
```

### What to Expect

  * The script will print status updates to the console, showing the loss for **Stage A (GNN)** and **Stage B (CNN)**.
  * Once training is complete, the final trained model parameters will be saved to `results/final_paper_model.jls`.
  * Finally, a series of evaluation plots (depth profiles, radial profiles, and event heatmaps) will be generated and displayed.

### Hyperparameters

All major hyperparameters (learning rates, weights, step counts) are configured as keyword arguments inside `scripts/train.jl`.

-----

## 📂 Project Structure

A brief overview of the repository layout:

```
Hybrid-Calo-UDE/
├── .gitignore          # Ignores logs and results/ content
├── Project.toml        # List of Julia dependencies
├── Manifest.toml       # Exact dependency versions for reproducibility
├── README.md           # This file
│
├── data/
│   └── (Your dataset file goes here)
│
├── results/
│   └── .gitignore      # Keeps this folder but ignores models/plots
│
├── scripts/
│   └── train.jl        # Main executable script to run the training
│
└── src/
    ├── ECalUDE.jl      # Main module file, includes all sub-files
    ├── data.jl         # Data loading, geometry init, physics profiles
    ├── models.jl       # Definitions for Stage A (GNN) and Stage B (CNN)
    ├── training.jl     # Loss functions and forward-pass logic
    └── utils.jl        # Helper functions (gradient clipping, etc.)
```

```
```
