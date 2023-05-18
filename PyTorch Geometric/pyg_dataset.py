import random
import torch
import networkx as nx
from tqdm import tqdm
import torch_geometric.utils as pygUtils
import matplotlib
from pathlib import Path
from torch_geometric.data import InMemoryDataset, download_url
from matplotlib import pyplot as plt

import Utilities.utils as utils


# TODO: Automatically detect changes in the raw data and reprocess
# TODO: Automatically detect changes in the features and reprocess

class MIDSdataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.raw_included_subdirs = kwargs.get('raw_included_subdirs', None)
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_dir = Path(self.raw_dir)
        if self.raw_included_subdirs is None:
            return list(raw_dir.glob('**/*.txt'))
        
        r = []
        for subdir in self.raw_included_subdirs:
            r.extend(list(Path(raw_dir / subdir).glob('**/*.txt')))
        return r
        

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
    #     return

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        with tqdm(total=len(self.raw_file_names)) as pbar:
            for graph_file in self.raw_file_names:
                data = self.make_data(graph_file)
                data_list.extend(data)
                pbar.update(1)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def make_data(self, graph_file):
        G = nx.read_edgelist(graph_file, nodetype=int)
        
        # Add node degree as a feature.
        for node in G.nodes():
            G.nodes[node]['degree'] = G.degree(node)
        
        # Add degree centrality as a feature.
        degree_cent = nx.degree_centrality(G)
        for node in G.nodes():
            G.nodes[node]['degree_centrality'] = degree_cent[node]
            
        # Add betweenness centrality as a feature.
        between_cent = nx.betweenness_centrality(G)
        for node in G.nodes():
            G.nodes[node]['betweenness_centrality'] = between_cent[node]
            
        torch_G = pygUtils.from_networkx(G, group_node_attrs=['degree', 'degree_centrality', 'betweenness_centrality'])
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
        nx.draw(G, with_labels=True, node_color=data.y, cmap=matplotlib.colormaps['bwr'])
        plt.show()
        

def inspect_dataset(dataset, num_graphs=1):
    for i in random.sample(range(len(dataset)), num_graphs):    
        data = dataset[i]  # Get a random graph object

        print()
        print(data)
        print('=============================================================')

        # Gather some statistics about the first graph.
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
        
        MIDSdataset.visualize_data(data)


def main():
    data = Path.cwd() / 'Dataset'
    raw_included_subdirs = None
    
    dataset = MIDSdataset(data, raw_included_subdirs=raw_included_subdirs)
    
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    inspect_dataset(dataset, num_graphs=0)
    

if __name__ == '__main__':
    main()