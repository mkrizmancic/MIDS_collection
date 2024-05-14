import random
from pathlib import Path

import codetiming
import matplotlib
import networkx as nx
import numpy as np
import torch
import torch.multiprocessing as torch_mp
import torch_geometric.utils as pygUtils
import Utilities.utils as utils
import yaml
from matplotlib import pyplot as plt
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

torch_mp.set_sharing_strategy('file_system')

raw_download_url = "https://github.com/mkrizmancic/MIDS_collection/raw/master/PyTorch%20Geometric/Dataset/raw_data.zip"


class MIDSdataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.selected_graph_sizes = kwargs.get("selected_graph_sizes", None)
        self.selected_graph_files = (
            [f"graphs{size:02d}.txt" for size in self.selected_graph_sizes]
            if self.selected_graph_sizes is not None
            else None
        )

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """
        Return a list of all raw files in the dataset.

        This method has two jobs. The returned list with raw files is compared
        with the files currently in raw directory. Files that are missing are
        automatically downloaded using download method. The second job is to
        return the list of raw file names that will be used in the process
        method.
        """
        raw_dir = Path(self.raw_dir)
        raw_files = []
        with open(raw_dir.parent / "file_list.yaml", "r") as file:
            raw_file_list = sorted(yaml.safe_load(file))
            for filename in raw_file_list:
                if self.selected_graph_files is None or filename in self.selected_graph_files:
                    raw_files.append(filename)

        return raw_files

    @property
    def processed_file_names(self):
        """
        Return a list of all processed files in the dataset.

        If a processed file is missing, it will be automatically created using
        the process method.

        That means that if you want to reprocess the data, you need to delete
        the processed files and reimport the dataset.
        """
        # TODO: Automatically detect changes in the dataset.
        #       We could come up with a namig scheme that will differentiate
        #       which graph families (and/or sizes) and features were used to
        #       generate the dataset. This way, we could detect changes and
        #       reprocess the dataset when needed.
        return ["data.pt"]

    def download(self):
        """Automatically download raw files if missing."""
        # TODO: Should check and download only missing files.

        zip_file = Path(self.root) / "raw_data.zip"

        # Delete the exising zip file.
        zip_file.unlink(missing_ok=True)

        # Download the raw files using the helper function.
        download_url(raw_download_url, self.root, filename="raw_data.zip")

        # Unzip the downloaded files.
        extract_zip(str(zip_file.resolve()), self.raw_dir)

    def process(self):
        """Process the raw files into a graph dataset."""
        # Read data into huge `Data` list.
        raw_data_list = []
        print("  Loading data from files, computing features and labels...")
        with tqdm(total=len(self.raw_file_names)) as pbar:
            for graph_file in self.raw_file_names:
                edge_lists_in_graph_file = []
                with open(Path(self.raw_dir) / graph_file, "r") as f:
                    for line in f.readlines():
                        graph_num, edge_list = line.split(": ")
                        edge_lists_in_graph_file.append(edge_list)

                result = process_map(MIDSdataset.make_data, edge_lists_in_graph_file, chunksize=100, max_workers=8)
                unpacked_result = [datapoint for single_graph_data in result for datapoint in single_graph_data]
                raw_data_list.extend(unpacked_result)
                pbar.update(1)

        print("  Converting data to PyG format...")
        torch_data_list = []
        for raw_data in tqdm(raw_data_list):
            torch_data_list.append(MIDSdataset.make_torch(raw_data))
        # -> Approach with tqdm multiprocessing wrapper.
        # torch_data_list = process_map(MIDSdataset.make_torch, raw_data_list, chunksize=100)
        # -> Approach with torch multiprocessing and manual tqdm.
        # m_raw_data_list = np.arange(len(raw_data_list))
        # with torch_mp.Pool(8, maxtasksperchild=10) as pool:
        #     result = list(tqdm(pool.imap_unordered(MIDSdataset.make_torch, m_raw_data_list, chunksize=10), total=len(raw_data_list)))
        # torch_data_list = [datapoint for single_graph_data in result for datapoint in single_graph_data]
        # -> Approach with torch multiprocessing and manual tqdm and shared data structure.
        # data_indices = np.arange(len(MIDSdataset.raw_data_list))
        # with torch_mp.Pool(8) as pool:
        #     result = list(tqdm(pool.imap(MIDSdataset.make_torch, data_indices, chunksize=100), total=len(data_indices)))
        # torch_data_list = [datapoint for single_graph_data in result for datapoint in single_graph_data]
        # -> All above approaches fail because of memory issues. Essentially, the problem is that each time an object
        #    in the input list is accessed, its copy is written to memory, taking up 8x the necessary memory.
        #    See more below.
        #    * https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
        #    * https://github.com/pytorch/pytorch/issues/13246#issuecomment-445446603
        #    * https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        #    * https://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-multiprocessing/5550156#5550156
        #    * https://stackoverflow.com/questions/10721915/shared-memory-objects-in-multiprocessing/10724332#10724332

        if self.pre_filter is not None:
            torch_data_list = [data for data in torch_data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            torch_data_list = [self.pre_transform(data) for data in torch_data_list]

        data, slices = self.collate(torch_data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def make_data(edge_list):
        """Create a PyG data object from a graph file."""
        # Load the graph from the file.
        # We assume that the index of the nodes is the same as the node label.
        # By default, networkx adds the nodes in the order they are found in the
        # edgelist. For example, if the edgelist is [(1, 3), (2, 3)], the order
        # of the nodes will be [1, 3, 2]. This interferes with the adjacency
        # matrix ordering.
        edges = [tuple(map(lambda x: int(x) - 1, edge.split(','))) for edge in edge_list.split('; ')]

        G_init = nx.from_edgelist(edges)
        G = nx.Graph()
        G.add_nodes_from(sorted(G_init.nodes))
        G.add_edges_from(G_init.edges)

        # Define features in use.
        feature_functions = {
            "degree": G.degree,
            "degree_centrality": nx.degree_centrality(G),
            "betweenness_centrality": nx.betweenness_centrality(G),
            "disjunction_value": utils.disjunction_value(G),
        }

        # Compute and add features to the nodes in the graph.
        for node in G.nodes():
            for feature in feature_functions:
                G.nodes[node][feature] = feature_functions[feature][node]

        # for node in G.nodes():
        #     G.nodes[node]["degree"] = G.degree(node)

        # degree_cent = nx.degree_centrality(G)
        # for node in G.nodes():
        #     G.nodes[node]["degree_centrality"] = degree_cent[node]

        # between_cent = nx.betweenness_centrality(G)
        # for node in G.nodes():
        #     G.nodes[node]["betweenness_centrality"] = between_cent[node]

        true_labels = MIDSdataset.get_labels(utils.find_MIDS(G), G.number_of_nodes())

        return [(G, label) for label in true_labels]

    @staticmethod
    def make_torch(raw_data):
        G, label = raw_data
        torch_G = pygUtils.from_networkx(G, group_node_attrs="all")
        torch_G.y = torch.from_numpy(label)
        return torch_G

    @staticmethod
    def get_labels(mids, num_nodes):
        # Encode found cliques as support vectors.
        for i, nodes in enumerate(mids):
            mids[i] = np.zeros(num_nodes)
            mids[i][nodes] = 1

        return mids

    @staticmethod
    def visualize_data(data):
        G = pygUtils.to_networkx(data, to_undirected=True)
        nx.draw(G, with_labels=True, node_color=data.y, cmap=matplotlib.colormaps["bwr"])
        plt.show()


def inspect_dataset(dataset, num_graphs=1):
    for i in random.sample(range(len(dataset)), num_graphs):
        data = dataset[i]  # Get a random graph object

        print()
        print(data)
        print("=============================================================")

        # Gather some statistics about the first graph.
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")

        MIDSdataset.visualize_data(data)


def main():
    root = Path(__file__).parent / "New Dataset"
    selected_graph_sizes = None

    with codetiming.Timer():
        dataset = MIDSdataset(root, selected_graph_sizes=selected_graph_sizes)

    print()
    print(f"Dataset: {dataset}:")
    print("====================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")

    inspect_dataset(dataset, num_graphs=1)


if __name__ == "__main__":
    main()
