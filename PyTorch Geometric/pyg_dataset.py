import random
from pathlib import Path

import codetiming
import matplotlib
import networkx as nx
import torch
import torch_geometric.utils as pygUtils
import Utilities.utils as utils
from matplotlib import pyplot as plt
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from tqdm import tqdm

from my_graphs_dataset import GraphLoader

raw_download_url = "https://github.com/mkrizmancic/MIDS_collection/raw/master/PyTorch%20Geometric/Dataset/raw_data.zip"


class MIDSdataset(InMemoryDataset):
    def __init__(self, root, loader: GraphLoader, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.loader = loader

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return str(self.loader.raw_files_dir.resolve())

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
        return self.loader.raw_file_names

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
        data_list = []
        with tqdm(self.raw_file_names) as files_w_progress:
            for graph_file in files_w_progress:
                files_w_progress.set_description(f"Processing {graph_file}")
                with open(Path(self.raw_dir) / graph_file, "r") as f:
                    for line in tqdm(f.readlines()):
                        graph = self.loader.load_graph(line)  # TODO: Make everything load from the loader and filter the number of loaded graphs.
                        data = self.make_data(graph)
                        data_list.extend(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def make_data(self, G):
        """Create a PyG data object from a graph object."""

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

        torch_G = pygUtils.from_networkx(G, group_node_attrs=list(feature_functions.keys()))
        true_labels = MIDSdataset.get_labels(utils.find_MIDS(G), G.number_of_nodes())
        data = []
        for labels in true_labels:
            data.append(torch_G.clone())
            data[-1].y = labels

        return data

    @staticmethod
    def get_labels(mids, num_nodes):
        # Encode found cliques as support vectors.
        for i, nodes in enumerate(mids):
            mids[i] = torch.zeros(num_nodes)
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
    root = Path(__file__).parent / "Dataset"
    selected_graph_sizes = None
    loader = GraphLoader(selection=selected_graph_sizes)

    with codetiming.Timer():
        dataset = MIDSdataset(root, loader)

    print()
    print(f"Dataset: {dataset}:")
    print("====================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")

    inspect_dataset(dataset, num_graphs=1)


if __name__ == "__main__":
    main()
