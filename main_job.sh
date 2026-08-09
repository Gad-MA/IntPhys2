#!/bin/bash
# =============================================================================
# main_job.sh — IntPhys2 PSI-0.5 benchmark launcher for the RCS cluster
#
# Submitted via Open OnDemand Job Composer.
# This script:
#   1. Clones IntPhys2 from GitHub (or pulls if already cloned)
#   2. Creates / reuses a Python venv and installs requirements
#   3. Launches main_distributed.py, which uses submitit to submit the real
#      6-node GPU evaluation job into SLURM.
# =============================================================================

#SBATCH --job-name=psi_intphys2
#SBATCH --partition=gpu
#SBATCH --account=grp-asem.a.abdelaziz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/mohamed.gad04/logs/psi_launcher_%j.out
#SBATCH --error=/home/mohamed.gad04/logs/psi_launcher_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_URL="https://github.com/Gad-MA/IntPhys2.git"
PROJECT_ROOT="$HOME/IntPhys2"
EVAL_ROOT="$PROJECT_ROOT/prediction_evals"
VENV_DIR="$HOME/venvs/intphys2"
CONFIG="$EVAL_ROOT/evals/intphys2/configs/psi.yaml"
SUBMITIT_LOGS="$EVAL_ROOT/logs/submitit"
RESULTS_DIR="$EVAL_ROOT/results/psi0_5"

mkdir -p "$HOME/logs"

echo "=========================================="
echo " IntPhys2 PSI-0.5 benchmark launcher"
echo " Cluster  : rcs"
echo " Project  : $PROJECT_ROOT"
echo " Venv     : $VENV_DIR"
echo " Config   : $CONFIG"
echo " Logs     : $SUBMITIT_LOGS"
echo " Results  : $RESULTS_DIR"
echo " Started  : $(date)"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. Clone or pull the repository
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] Setting up repository..."

if [ -d "$PROJECT_ROOT/.git" ]; then
    echo "  Repo already exists — pulling latest changes..."
    git -C "$PROJECT_ROOT" pull --ff-only
else
    echo "  Cloning $REPO_URL → $PROJECT_ROOT ..."
    git clone "$REPO_URL" "$PROJECT_ROOT"
fi

echo "  Repo is up to date. (commit: $(git -C "$PROJECT_ROOT" rev-parse --short HEAD))"

# ---------------------------------------------------------------------------
# 2. Create / reuse the virtual environment
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Setting up Python environment..."

if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "  Venv already exists — reusing."
fi

# Activate the venv for the remainder of this script
source "$VENV_DIR/bin/activate"
echo "  Python: $(python --version)  |  pip: $(pip --version | awk '{print $2}')"

# Always (re-)install requirements so any new deps are picked up after a pull
echo "  Installing / updating requirements..."
pip install --upgrade pip --quiet
pip install -r "$EVAL_ROOT/requirements.txt" --quiet
echo "  Requirements satisfied."

# ---------------------------------------------------------------------------
# 3. Launch the distributed evaluation
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Submitting evaluation job via submitit..."
echo "  submitit will request 6 nodes × 1 GPU each (~72 h timeout)."
echo ""

mkdir -p "$SUBMITIT_LOGS" "$RESULTS_DIR"
cd "$EVAL_ROOT"

python -m evals.main_distributed \
    --fname   "$CONFIG"              \
    --folder  "$SUBMITIT_LOGS"       \
    --partition gpu                  \
    --account grp-asem.a.abdelaziz  \
    --qos     normal                 \
    --time    4300

echo ""
echo "=========================================="
echo " Launcher finished at $(date)."
echo " Real eval job IDs are in: $SUBMITIT_LOGS"
echo "=========================================="
