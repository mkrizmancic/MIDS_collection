## About
This package creates the dataset for the Minimum Independent Dominating Set (MIDS) problem in PyTorch Geometric (PyG) format. The dataset consists of all possible variations of MIDS for all unique graphs with size (number of nodes) between 3 and 10 inclusive.

The number of unique graphs for each size is as follows ([more here](https://oeis.org/A001349/list)):
| Index | 3 | 4 | 5 | 6 | 7 |  8  |   9  |   10   |
|-------|---|---|---|---|---|-----|------|--------|
| Value | 2 | 6 | 21|112|853|11117|261080|11716571|


Since the number of graphs for sizes 9 and 10 is very large, it is recommended to limit the number of loaded graphs.

## Preparation
### Setting up the environment
To avoid cluttering your environment with potentially incompatible packages, it is recommended to create a virtual environment and install packages within it. `setup_environment.sh` can do that for you. You can choose to install the CPU or GPU (CUDA) version of the library. Simple comment (out) the corresponding lines in the script. By default, the script will install the CPU version.

Position yourself in the `PyTorch Geometric` directory. From it, run `bash Utilities/setup_environment.sh`. After that, you should be ready to generate the dataset.

### Getting data
This dataset uses another "base" dataset that contains the raw files with graph specifications. The dataset is available [here](https://github.com/mkrizmancic/my_graphs_dataset). Follow the instructions in the dataset repository to set it up.

Once set up, you can make a list of files that will be used in this dataset generation. The file list is used to monitor if all the necessary files are present and to regenerate the dataset if needed. To make the file list, run the following command from the `PyTorch Geometric/Dataset` directory:

```python3 -m my_graphs_dataset.file_list_snapshot```


## Usage
Running the `pyg_dataset.py` will do the following:
1. Read the raw files from the base dataset.
1. Compute the features for all nodes in each graph from the dataset.
1. Save the result in PyG-readable format to the disk.
1. Print out the information about the dataset.
1. _Optionally_ print more detailed information about the randomly selected graph(s) from the dataset and plot them to visualize the MIDS. (To enable this feature, change the `num_graphs` argument in the `inspect_dataset` function call.)

Alternatively, you can import the dataset as you would normally do in PyG tutorials.

### Dataset features
All features are calculated with standard networkx library functions and can thus be easily extended.
