#!/bin/bash

# 1. Install dependencies
echo "Installing required libraries..."
pip install numpy scipy pandas matplotlib tqdm

# 2. Generate the datasets (Required for main.py)
echo "Step 1: Generating datasets..."
python generate_datasets.py

# 3. Run the Active POR experiments
echo "Step 2: Running Active Preference-based Ordinal Regression..."
python main.py

echo "Process complete. Results should be generated."
