#!/bin/bash

#SBATCH --partition=juno --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time="7-0"
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --job-name="temporal-policies"
#SBATCH --output=logs/juno-%j.out
#SBATCH --mail-user="cagia@stanford.edu"
#SBATCH --mail-type=FAIL,REQUEUE

# List out some useful information.
echo "SLURM_JOBID=${SLURM_JOBID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "Working directory: ${SLURM_SUBMIT_DIR}"
echo ""
echo ${1}
echo ""

# Activate Pyenv and run the command in Pipenv.
export PYENV_ROOT="${HOME}/.pyenv"
export PATH="${PYENV_ROOT}/bin:${PATH}"
if which pyenv > /dev/null; then
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)";
fi
pipenv run ${1}

# Send an email upon completion.
MAIL_SUBJECT="'SLURM Job_id=${SLURM_JOBID} Log'"
MAIL_FILE="$(pwd -P)/logs/juno-${SLURM_JOBID}.out"
MAIL_CMD="mail -s ${MAIL_SUBJECT} cagia@stanford.edu < ${MAIL_FILE}"
~/ssh-bohg-ws-12.py "${MAIL_CMD}"
