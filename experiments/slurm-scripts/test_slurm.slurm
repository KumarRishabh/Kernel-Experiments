#!/bin/bash
#SBATCH --job-name=test_job              # Job name
#SBATCH --partition=cpu                 # Use 'gpu' if testing GPU jobs
#SBATCH --cpus-per-task=2               # Number of CPU cores
#SBATCH --mem=4G                         # Memory required
#SBATCH --time=00:10:00                  # Short runtime (10 mins)
#SBATCH --output=logs/test_%j.out        # STDOUT log
#SBATCH --error=logs/test_%j.err         # STDERR log

# ------------------------
# Load required modules
# ------------------------
module load python/3.9.12

# ------------------------
# Print environment info
# ------------------------
echo "Running on host: $(hostname)"
echo "Date: $(date)"
echo "Current directory: $(pwd)"

# ------------------------
# Create a test Python script
# ------------------------
cat << EOF > test_script.py
print("Hello from SLURM!")
import platform
print("Python version:", platform.python_version())
EOF

# ------------------------
# Run the test script
# ------------------------
python ../test_scripts/python-slurm-test.py