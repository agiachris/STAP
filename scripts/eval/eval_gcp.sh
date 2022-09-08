#!/bin/bash

#SBATCH --partition=eval --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time="7-0"
#SBATCH --mem=6G
#SBATCH --gres=gpu:1
#SBATCH --job-name="temporal-policies"
#SBATCH --output=logs/gcp-%j.out
#SBATCH --mail-user="takatoki@stanford.edu"
#SBATCH --mail-type=FAIL,REQUEUE

# List out some useful information.
echo "SLURM_JOBID=${SLURM_JOBID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo ""
echo ${1}
echo ""

export PYENV_ROOT="${HOME}/.pyenv"
export PATH="${PYENV_ROOT}/bin:${PATH}"
if which pyenv > /dev/null; then
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)";
fi
pipenv run ${1}

# MAIL_SUBJECT="'SLURM Job_id=${SLURM_JOBID} Log'"
# MAIL_FILE="$(pwd -P)/logs/gcp-${SLURM_JOBID}.out"
# MAIL_CMD="mail -s ${MAIL_SUBJECT} takatoki@stanford.edu < ${MAIL_FILE}"
