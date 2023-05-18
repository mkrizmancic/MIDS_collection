## About
This package creates the dataset for Minimum Independent Dominating Set (MIDS) problem in PyTorch Geometric (PyG) format. The dataset consists of all possible variations of MIDS for all unique graphs with size (number of nodes) between 3 and 8 inclusive.

## Preparation
### Getting data
If you have `git lfs` installed, the zip file with all the graphs will download automatically when you `git clone` or `git pull`. To install and set up `git lfs`, follow this [tutorial](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=linux).

When you have the zip, unpack it to `Dataset/raw/`.

### Setting up environment
To avoid clutering your environment with potentially incompatible packages, it is recommended to create a virtual environment and install packages within it. `setup_environment.sh` can do that for you.

Position yourself in `PyTorch Geometric` directory. From it, run `bash Utilities/setup_environment.sh`. After that, you should be ready to generate the dataset.

## Usage
Running the `pyg_dataset.py` will do the following:
* read the raw files from `raw` directory,
* compute the features for all nodes in each graph from the dataset,
* save the result in PyG-readable format to the disk,
* print out the information about the dataset,
* _optionally_ print more detailed information about randomly selected graph(s) from the dataset and plot them to visualize the MIDS. (To enable this feature, change the `num_graphs` argument in the `inspect_dataset` function call.)

Alternatively, you can import the dataset as you would normally do in PyG tutorials.

### Dataset features
Currently, the nodes have the following features:
* degree
* degree_centrality
* betweenness_centrality

All features are calculate with standard networkx library functions and can thus be easiliy extended.
