#!/usr/bin/env bash

# Create or reuse the virtual environment if it doesn't already exist
if [ ! -d "maket_env" ]; then
    python -m venv market_env
fi

source market_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
