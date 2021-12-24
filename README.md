# Research Lightning

This is a lightweight research framework designed to quickly implement and test deep learning algorithms in pytorch. This Readme describes the general structure of the framework and how to use each of its components. Here are some key features of the repository:
* Support for supervised learning algorithms
* Tensorboard logging
* Support for reinforcement learning algorithms
* Hardware stripping on both local machines and SLURM

I would suggest reading through all the documentation. If you use this package, please cite it appropriately as described in the [Usage](#Usage) section.

## Getting Started
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

If there are updates to the template, you can later pull them by running `git pull template <branch of research-lightning you want to use>`.

## Code Design
TODO

## Usage
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
