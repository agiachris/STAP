# Make sure we have the conda environment set up.
CONDA_PATH=~/miniconda3/bin/activate
ENV_NAME=research
REPO_PATH=path/to/your/repo
USE_MUJOCO_PY=false

# Setup Conda
source $CONDA_PATH
conda activate $ENV_NAME 
cd $REPO_PATH

unset DISPLAY # Make sure display is not set or it will prevent scripts from running in headless mode.

if $USE_MUJOCO_PY; then
    echo "Using mujoco_py"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
fi

# Try to import CUDA if we can, also add mujoco rendering backends. OK if unused.
if [ -d "/usr/local/cuda-11.1" ]; then
    export PATH=/usr/local/cuda-11.1/bin:$PATH
    export MUJOCO_GL="egl" 
elif [ -d "/usr/local/cuda-11.0" ]; then
    export PATH=/usr/local/cuda-11.0/bin:$PATH
    export MUJOCO_GL="egl" 
elif [ -d "/usr/local/cuda-10.2" ]; then
    export PATH=/usr/local/cuda-10.2/bin:$PATH
    export MUJOCO_GL="egl" 
else
    echo "Could not find a CUDA version, using CPU."
    export MUJOCO_GL="osmesa"
fi
