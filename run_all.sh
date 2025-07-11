#!/bin/bash

echo "Starting HMLR NLP pipeline..."

# Step 1: Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt || { echo " Failed to install dependencies."; exit 1; }

# Step 2: Run the main pipeline
echo " Running pipeline..."
PYTHONPATH=. python main.py || { echo " Pipeline failed."; exit 1; }

# Step 3: Run tests
echo "Running unit tests..."
PYTHONPATH=. pytest tests/ || { echo " Tests failed."; exit 1; }

# Step 4: List output files
echo " Outputs directory:"
ls -lh outputs/

echo "Pipeline finished successfully!"
