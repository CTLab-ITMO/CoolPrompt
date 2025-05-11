#!/bin/bash

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not installed. Please install it and try again."
    exit 1
fi

# Create a virtual environment in the current directory
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found in the current directory."
    deactivate
    exit 1
fi

# Install the requirements
pip install -r requirements.txt

echo "Virtual environment setup complete and requirements installed."

# Deactivate the virtual environment
deactivate