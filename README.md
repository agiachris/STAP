# Research Lightning

This is a lightweight research framework designed to quickly implement and test deep learning algorithms in pytorch. This Readme describes the general structure of the framework and how to use each of its components. Here are some key features of the repository:
* Support for supervised learning algorithms
* Tensorboard logging
* Support for reinforcement learning algorithms
* Hardware stripping on both local machines and SLURM

I would suggest reading through all the documentation. If you use this package, please cite it appropriately as described in the [Usage](#Usage) section.

## Installation

The repository is split into multiple branches, each with a different purpose. Each branch contains implementations of standard algorithms and datasets. There are currently three main branches: main, image, and rl. Choose the branch based on the default implementations or examples you want included.

First, create an github repository online. DO NOT initialize the repoistory with a `README`, `.gitignore`, or `lisence`. We are going to set up a repository with two remotes, one being the new repo you just created to track your project, and the other being the template repository.

```
mkdir <your project name>
git init
git remote add template https://github.com/jhejna/research-lightning
git remote set-url --push template no_push
git pull template <branch of research-lightning you want to use>
git branch -M main
git remote add origin https://github.com/<your-username>/<your project name>
git push -u origin main
```
You should now have setup a github repository with the research-lightning base. If there are updates to the template, you can later pull them by running `git pull template <branch of research-lightning you want to use>`.

After setting up the repo, there are a few steps before you can get started:
1. Edit `environment_cpu.yaml` and `environment_gpu.yaml` as desired to include any additional dependencies via conda or pip. You can also change the name if desired.
2. Create the conda environment using `conda env create -f environment_<cpu or gpu>.yaml`.
3. Install the research package via `pip install -e research`.
4. Modify the `setup_shell.sh` script by updated the appropriate values as needed. The `setup_shell.sh` script should load the environment, move the shell to the repository directory, and additionally setup any external dependencies. You can add any extra code here.

Other default configuration values for the sweepers, particularly slurm, can be modified at the header of `tools/run_slurm.py`.

## Usage
You should be able to activate the development enviornment by running `. path/to/setup_shell.sh`. This is the same environment that will be activated when running jobs on SLURM.

### Adding Your Own Research
Training is managed by five types of objects: algorithms, networks, environments, datasets, and processors. Each has their own sub-package located within the `research` directory. Each `__init__.py` file in the subpackage exposes classes to the global trainer.

All algorithms can be implemented by extending the `Algorithm` base class found in `algs/base.py`. Algorithms take in all parameters, including the class types of other objects, and run training. Most handling is already implemented, but `setup` methods may be overridden for specific functionality.

Algorithms are required to have an `environment`, which for RL algorithms can be a gym environment, or for supervised learning can simply be a object containing gym spaces that dictate the models input and output spaces like the `Empty` environment implemented in `envs/empty.py`. 

Networks must extend `nn.Module` and contain the entirety of your model. If multiple models are required, write a wrapper `network` class that contains both of them. Examples of this can be seen in the actor critic algorithms in the `rl` branch. All networks take in an observation (input) and action (output) space.

All data functionality is managed through pytorch's `Dataset` and `Dataloader` classes. The `dataset` submodule should contain implementations of datasets.

Finally `processors` are applied to batches of data before they are fed to networks. It's often useful to have data processing outside of the Dataset object as it can be done in parallel instead of on individual samples. 

All training is handled through `utils.trainer`, which assembles all of the specified components and calls `Algorithm.train`. Components are specified via config yaml files that are parsed before training. The config parser supports importing arbitrary python objects by specifying a list of strings starting with the "import" keyword, for example `activation: ["import", "torch.nn", "ReLU"]`.

Examples of how each of these different systems can be used can be found in the `vision` and `rl` branches that contain default implementations. The recommend steps for implementing your own method are as follows:

1. Implement the environment, which will determine the input/output spaces in `envs`. If its a gym environment, you can register it.
2. Implement the dataset that is built on top of the environnment.
3. Implement any processors your might have.
4. Implement the network
5. Implement the algorithm by extending the base class, filling out the `_train_step` method (and optionally others).
6. Create a config, see `configs/example.yaml` for an idea of how it works.

### Launching jobs
To train a model, simply run `python scripts/train.py --config path/to/config --path path/to/save/folder`

Results can be viewed on tensorboard.

The `tools` folder contains simple tools for launching job sweeps locally or on a SLURM cluster. The tools work for every script, but have special features for `scripts/train.py`. 

To launch any job via a script in the `tools` folder, use the `--entry-point <path to script>` argument to specify the path to the target script (`scripts/train.py`) by default and the `--arguments <arg1>=<value1>  <arg2>=<value2> ..  <argk>=<valuek>` to specify the arguments for the script. Multiple different jobs can be stacked. For example, `--arguments` can be provided more than once to specify different sets of arguments. The `--seeds-per-job` argument lets you run multiple seeds for a given entry-point, but the entry-point script will need to accept a `--seed` argument.

#### Local Jobs
Launching jobs locally can easily be done by specifying `--cpus` in the same manner you would for `taskset` and `--gpus` via nvidia devices. Note that multi-gpu training is not yet supported. The script will automatically balance jobs on the provided hardware. Here is an example:
```
python tools/run_local.py scripts/my_custom_script.py --cpus 0-8 --gpus 0 1 --seeds-per-job 2 --arguments <arg1>=<value1>  <arg2>=<value2>
```
This will run one job on cores 0-3 with GPU 0 and one job on cpus 4-7 with GPU 1. 

#### SLURM
Launching jobs on SLURM is done via the `tools/run_slurm.py` script. In addition to the base arguments for job launching, the slurm script takes several additional arguments for slurm. Here is an example command that includes all of the required arguments and launches training jobs from `scripts/train.py`. Additional optional arguments can be found in the `tools/run_slurm.py` file.
```
python tools/run_slurm.py --partition <partition> --cpus 8 --gpus 1080ti:1 --mem 16G --job-name example --arguments config=configs/example.yaml path=../output/test
```
The `gpu` argument takes in the GRES specification of the GPU resource. One unfortunate problem with GRES it doesn't allow sharing GPUs between slurm jobs. This often means that if you want to run a small model that only consumes, say 25% of the GPUs max FLOPS, everyone else on the cluster will still be blocked from using the GPU. The `--jobs-per-instance` argument allows you to train multiple models on the same SLURM node in parallel on the same GPU! You just need to make sure to specify enough CPU and memory resources to run both at once. Doing so drastically saves GPU resources on the cluster if you are running parameter sweeps.

#### Using the Sweeper
The default training script `scripts/train.py` also supports parameter sweeps. Parameter sweeps are specified using `json` files. Any of the scripts in `tools` will automatically detect that a sweep is being run based on the `entry-point` and type of config file (json) being specified. An example sweep file is found on the `vision` branch. Keys in the json file are specified via strings, with periods to separate nested structions, and values are provided as lists. For example to specify a learning rate sweep, one would add `"optim_kwargs.lr" : [0.01, 0.001]`. 

There are two special keys. The first `"base"` is required and specifies the path to the base config that will be modified by the sweep file. The second, `"paired_keys"` allows you to pair the values of differnet parameters in the sweep. 

## Included Implementations
This repo contains high quality implementations of different algorithms on the `rl` and `vision` branches. Here are some features of each of the branches:

### Vision

* Generic Image Classification
* Wrapper to use arbitrary torchvision datasets

### RL

* ReplayBuffer class that elegantly handles both parallel and serial Dataloading as well as Dict observation and action spaces. From my benchmarking this is one of the fastest implementations I know. It borrows heavily from [DrQv2](https://github.com/facebookresearch/drqv2), but can sample entire batches of data at once to avoid serial collation. For TD3 this lead to around a 15% speed increase.
* Gym wrappers for `dm_control`, matching those in [DrQv2](https://github.com/facebookresearch/drqv2) and [pytorch_sac](https://github.com/denisyarats/pytorch_sac).
* High quality TD3 implementation
* SAC implementation, borrowing closely from [pytorch_sac](https://github.com/denisyarats/pytorch_sac).

## License
This framework has an MIT license as found in the [LICENSE](LICENSE) file.

If you use this package, please cite this repository. Here is the associated Bibtex:
```
@misc{hejna2021research,
    title={Research Lightning: A lightweight package for Deep Learning Research},
    author={Donald J Hejna III},
    year={2021},
    publisher={GitHub},
    journal={GitHub Repository},
    howpublished = {\url{https://github.com/jhejna/research-lightning}}
}
```
