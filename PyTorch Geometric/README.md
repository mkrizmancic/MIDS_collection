## About
This package creates the dataset for the Minimum Independent Dominating Set (MIDS) problem in PyTorch Geometric (PyG) format. The dataset consists of all possible variations of MIDS for all unique graphs with size (number of nodes) between 3 and 8 inclusive.

## Preparation
### Getting data
Data will be downloaded automatically when running or importing the dataset.

### Setting up the environment
To avoid cluttering your environment with potentially incompatible packages, it is recommended to create a virtual environment and install packages within it. `setup_environment.sh` can do that for you. You can choose to install the CPU or GPU (CUDA) version of the library. Simple comment (out) the corresponding lines in the script. By default, the script will install the CPU version.

Position yourself in the `PyTorch Geometric` directory. From it, run `bash Utilities/setup_environment.sh`. After that, you should be ready to generate the dataset.

## Usage
Running the `pyg_dataset.py` will do the following:
1. Download raw files if they don't exist on disk.1
1. Read the raw files from the `raw` directory.
1. Compute the features for all nodes in each graph from the dataset.
1. Save the result in PyG-readable format to the disk.
1. Print out the information about the dataset.
1. _Optionally_ print more detailed information about the randomly selected graph(s) from the dataset and plot them to visualize the MIDS. (To enable this feature, change the `num_graphs` argument in the `inspect_dataset` function call.)

Alternatively, you can import the dataset as you would normally do in PyG tutorials.

### Dataset features
Currently, the nodes have the following features:
* degree
* degree_centrality
* betweenness_centrality

All features are calculated with standard networkx library functions and can thus be easily extended.
