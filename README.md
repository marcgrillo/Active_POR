Active Learning for Preference Ordering (Active_POR)

This project implements and benchmarks Bayesian and Optimization-based Active Learning strategies for Multi-Criteria Decision Analysis (MCDA). It simulates decision-makers (DMs) to evaluate how well different algorithms can learn preference models and rank alternatives using a minimal number of pairwise comparisons.

The passive learning algorithms used as a baseline are presented in the paper:

"Bayesian and optimization-based inference for preference learning in multi-criteria decision analysis"
European Journal of Operational Research, 2025.
Read the paper

🚀 Features

Preference Learning Algorithms:

BAYES: MCMC-based inference using Nested Sampling (dynesty).

FTRL: Optimization-based inference (Follow-the-Regularized-Leader) using MAP estimation and Laplace approximation (cvxpy, scipy).

Probability Models:

Linear (LIN): $P(a \succ b) = 0.5(1 + U(a) - U(b))$

Bradley-Terry (BT): $P(a \succ b) = \sigma(U(a) - U(b))$

Active Learning Strategies:

BALD (Bayesian Active Learning by Disagreement): Maximizes information gain.

US (Uncertainty Sampling): Queries the most uncertain pair.

Benchmarking Metrics:

POI/RAI: Probabilistic rankings.

AIOS: Average Individual Ordinal Stability (Stability of the best alternative).

ASPS: Average Simulated Preference Stability (Pairwise accuracy).

ASRS: Average Simulated Rank Stability.

📂 Project Structure

The codebase is organized into modular packages:

Active_POR/
├── main.py                  # Entry point for running benchmarks
├── generate_datasets.py     # Script to generate synthetic ground truth data
├── show_res.ipynb           # Notebook for visualizing results and metrics
│
├── common/                  # Shared utilities
│   └── utils.py             # Math helpers, I/O, and file handling
│
├── mcda/                    # Domain logic
│   └── models.py            # Piecewise-Linear Utility transformation logic
│
├── inference/               # Core Inference Engine
│   └── engine.py            # PreferenceSampler class (MCMC & Optimization logic)
│
└── experiments/             # Experiment orchestration
    ├── simulation.py        # Logic for a single decision-maker simulation
    ├── runner.py            # Batch processing for datasets
    └── metrics.py           # Calculation of ASRS, AIOS, POI, RAI


📦 Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

pip install numpy pandas scipy tqdm dynesty cvxpy matplotlib


Note: cvxpy is required for the robust convex optimization used in FTRL-BT.

🛠️ Usage

1. Generate Synthetic Datasets

First, generate the "Ground Truth" data (decision matrices and true utility functions). This script also saves the inconsistency parameters (lambda) used during generation.

python generate_datasets.py


Configuration (inside the file):

F1: Number of alternatives (e.g., [30])

F2: Number of criteria (e.g., [4])

target_inconsistency: Expected user error rate (e.g., 0.10).

2. Run Benchmarks

Run the active learning simulations. This reads the generated datasets and simulates the interaction loop.

python main.py


Configuration (inside main.py):

TARGET_METHODS: List of algorithms to run (e.g., ['BAYES_BT_BALD', 'FTRL_LIN_US']).

HM: Number of Human Models (tables) to process (e.g., 200).

3. Visualizing Results

A Jupyter Notebook is provided to visualize the computed metrics (ASRS, ASPS, AIOS).

Open show_res.ipynb in Jupyter/Lab:

jupyter notebook show_res.ipynb


This notebook contains functions to plot the evolution of stability metrics versus the number of queries, comparing "Active" vs "Regular" (Passive) learning.

🧠 Methodology Details

Models

The project assumes a Piecewise-Linear Marginal Utility model. The raw data is transformed into a feature matrix $X$ where the global utility is linear in parameters: $U(a) = \omega \cdot \phi(a)$.

Inference & Active Learning

Bayesian Approach: Maintains a full posterior distribution of weights $\omega$. Active queries are selected by estimating the mutual information between the model parameters and the query outcome (BALD).

FTRL Approach: Maintains a point estimate (MAP) and approximates the posterior locally as a Gaussian (Laplace Approximation) to compute acquisition scores efficiently without full MCMC sampling.

Inconsistency Simulation

User responses are simulated using a logistic model. The probability of a user flipping the "true" preference depends on the utility difference between options and a temperature parameter $\lambda$.
$$ P(\text{correct}) = \frac{1}{1 + e^{-\lambda (U_{winner} - U_{loser})}} $$
