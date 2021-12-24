# Make sure we have the conda environment set up.
{
    conda help
} || {
    source ~/miniconda3/bin/activate
} || {
    source ~/anaconda3/bin/activate
}

# Activate the conda environment
conda activate research 

# Change to the repository directory.
cd (dirname "$(realpath $0)")

unset DISPLAY # Make sure display is not set or it will prevent scripts from running in headless mode.

# Try to import CUDA if we can
if [ -d "/usr/local/cuda-11.1" ]
then
    export PATH=/usr/local/cuda-11.1/bin:$PATH
elif [ -d "/usr/local/cuda-11.0" ]
then
    export PATH=/usr/local/cuda-11.0/bin:$PATH
elif [ -d "/usr/local/cuda-10.2" ]
else
    echo "Could not find a CUDA version."
fi
