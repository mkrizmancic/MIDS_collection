import time
import numpy as np
import pandas as pd
import plotly.express as px
from milpMIDS import optimize
from my_graphs_dataset import GraphDataset, GraphType
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

random_dataset_description = {
    GraphType.BARABASI_ALBERT:      (100, range(10, 50+1, 5)),
    GraphType.ERDOS_RENYI:          (100, range(10, 50+1, 5)),
    GraphType.WATTS_STROGATZ:       (100, range(10, 50+1, 5)),
    GraphType.NEW_WATTS_STROGATZ:   (100, range(10, 50+1, 5)),
    GraphType.STOCH_BLOCK:          (100, range(10, 50+1, 5)),
    GraphType.REGULAR:              (100, range(10, 50+1, 5)),
    GraphType.CATERPILLAR:          (100, range(10, 50+1, 5)),
    GraphType.LOBSTER:              (100, range(10, 50+1, 5)),
    GraphType.POWER_TREE:           (100, range(10, 50+1, 5)),
    GraphType.FULL_K_TREE:  (100, range(10, 50+1, 5)),
    GraphType.LOLLIPOP:     (100, range(10, 50+1, 5)),
    GraphType.BARBELL:      (100, range(10, 50+1, 5)),
    GraphType.GRID:         (100, range(10, 50+1, 5)),
    GraphType.CAVEMAN:      (100, range(10, 50+1, 5)),
    GraphType.LADDER:       (100, range(10, 50+1, 5)),
    GraphType.LINE:         (100, range(10, 50+1, 5)),
    GraphType.STAR:         (100, range(10, 50+1, 5)),
    GraphType.CYCLE:        (100, range(10, 50+1, 5)),
    GraphType.WHEEL:        (100, range(10, 50+1, 5)),
}

small_dataset_description = {
    GraphType.BARABASI_ALBERT:      (10, range(10, 21)),
    GraphType.ERDOS_RENYI:          (10, range(10, 21)),
    GraphType.WATTS_STROGATZ:       (10, range(10, 21)),
    GraphType.NEW_WATTS_STROGATZ:   (10, range(10, 21))
}

datset_description = small_dataset_description


def process_graph(data):
    graph_type, graph_settings, pid = data
    loader = GraphDataset(selection={graph_type: graph_settings}, suppress_output=True)

    # Get all graphs first to determine total
    all_graphs = list(loader.graphs(raw=False))

    results = {}
    # Create a progress bar for this specific graph type
    for G in tqdm(all_graphs, desc=f"Processing {graph_type.name}", position=pid):
        solution, elapsed, dets = optimize(G, "MIDS", outputFlag=0)
        graph_size = len(G.nodes)
        if graph_size not in results:
            results[graph_size] = []
        results[graph_size].append(elapsed)
    return graph_type.name, results

def main():
    start = time.perf_counter()

    # Convert dataset description items to a list for process_map
    graph_type_settings = [(items[0], items[1], pid) for pid, items in enumerate(datset_description.items())]

    # Use process_map instead of Parallel
    parallel_results = process_map(
        process_graph,  # Unpack tuple of (graph_type, graph_settings)
        graph_type_settings,
        desc="Graph Types",
    )

    # Rest of your code remains the same
    results = {graph_type_name: size_results for graph_type_name, size_results in parallel_results}

    end = time.perf_counter()
    print(f"Processed {sum(len(size_results) for size_results in results.values())} graphs.")
    print(f"Total elapsed time: {end - start:.2f} seconds.")

    # Processing results into a DataFrame
    index_sizes = sorted(set(size for type_results in results.values() for size in type_results))
    df_avg = pd.DataFrame(index=index_sizes, columns=list(results.keys()))
    df_std = pd.DataFrame(index=index_sizes, columns=list(results.keys()))

    for graph_type, size_results in results.items():
        for size, times in size_results.items():
            df_avg.at[size, graph_type] = np.mean(times)
            df_std.at[size, graph_type] = np.std(times)

    # Convert to numeric values
    df_avg = df_avg.apply(pd.to_numeric)
    df_std = df_std.apply(pd.to_numeric)

    # Flatten data for hover text
    hover_data = []
    for size in df_avg.index:
        for graph_type in df_avg.columns:
            avg_time = df_avg.at[size, graph_type]
            std_dev = df_std.at[size, graph_type]
            hover_data.append((size, graph_type, f"{avg_time:.3f} ± {std_dev:.3f}"))

    hover_df = pd.DataFrame(hover_data, columns=['Graph Size', 'Graph Type', 'Hover Text'])
    hover_matrix = hover_df.pivot(index='Graph Size', columns='Graph Type', values='Hover Text')
    hover_matrix = hover_matrix.loc[:, list(df_avg.columns)]

    # Visualization
    fig = px.imshow(df_avg, labels={'x': "Graph Type", 'y': "Graph Size", 'color': "Avg Computation Time"},
                    x=df_avg.columns, y=df_avg.index, color_continuous_scale='Blues', text_auto=True)
    fig.update_traces(text=hover_matrix.values, hovertemplate="%{text}")
    fig.update_layout(title="Average Computation Time by Graph Type and Size (in ms)")
    fig.show()

    print("done")

if __name__ == '__main__':
    main()
