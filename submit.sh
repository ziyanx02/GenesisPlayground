#!/bin/bash

# List of environments
exp_name=${1:-None}

sbatch sbatch.sh "$exp_name"
