#!/bin/bash

# Wrapper script to ensure 'pace' conda environment is active
# This script should be sourced, not executed

# Check if running in a subshell (executed) vs sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "⚠️  This script should be sourced, not executed!"
    echo ""
    echo "Usage:"
    echo "  source activate_pace.sh"
    echo ""
    echo "Or simply run:"
    echo "  conda activate pace"
    exit 1
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Check if pace environment exists
if conda env list | grep -q "^pace "; then
    echo "Activating 'pace' conda environment..."
    conda activate pace
    
    if [[ "$CONDA_DEFAULT_ENV" == "pace" ]]; then
        echo "✓ 'pace' environment activated successfully"
        echo "Python: $(which python)"
        echo "Version: $(python --version)"
    else
        echo "✗ Failed to activate 'pace' environment"
        return 1
    fi
else
    echo "✗ 'pace' conda environment not found"
    echo ""
    echo "Create it with:"
    echo "  conda create -n pace python=3.10"
    echo "  conda activate pace"
    echo "  pip install -r requirements.txt"
    return 1
fi
