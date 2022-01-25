# Make sure we have the conda environment set up.
CONDA_PATH=~/miniconda3/bin/activate
ENV_NAME=temporal_policies
REPO_PATH=path/to/your/repo

# Setup Conda
source $CONDA_PATH
conda activate $ENV_NAME 
cd $REPO_PATH

unset DISPLAY # Make sure display is not set or it will prevent scripts from running in headless mode.

# Try to import CUDA if we can.
if [ -d "/usr/local/cuda-11.1" ]; then
    export PATH=/usr/local/cuda-11.1/bin:$PATH
elif [ -d "/usr/local/cuda-11.0" ]; then
    export PATH=/usr/local/cuda-11.0/bin:$PATH
elif [ -d "/usr/local/cuda-10.2" ]; then
    export PATH=/usr/local/cuda-10.2/bin:$PATH
fi
