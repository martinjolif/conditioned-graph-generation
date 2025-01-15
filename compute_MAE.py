import numpy as np
import math
from community import community_louvain
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast
from pathlib import Path
import json

def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x

def calculate_stats_graph(G):
    stats = []
    # Number of nodes
    num_nodes = handle_nan(float(G.number_of_nodes()))
    stats.append(num_nodes)
    # Number of edges
    num_edges = handle_nan(float(G.number_of_edges()))
    stats.append(num_edges)

    # Degree statistics
    degrees = [deg for node, deg in G.degree()]
    avg_degree = handle_nan(float(sum(degrees) / len(degrees)))
    stats.append(avg_degree)

    # Number of triangles
    triangles = nx.triangles(G)
    num_triangles = handle_nan(float(sum(triangles.values()) // 3))
    stats.append(num_triangles)

    # Global clustering coefficient
    global_clustering_coefficient = handle_nan(float(nx.transitivity(G)))
    stats.append(global_clustering_coefficient)
    # Maximum k-core
    max_k_core = handle_nan(float(max(nx.core_number(G).values())))
    stats.append(max_k_core)

    # calculate communities
    partition = community_louvain.best_partition(G)
    n_communities = handle_nan(float(len(set(partition.values()))))
    stats.append(n_communities)

    return stats

def sum_elements_per_column(matrix, dc):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    column_sums = [0] * num_cols

    for col in range(num_cols):
        for row in range(num_rows):
            column_sums[col] += matrix[row][col]

    res = []
    for col in range(num_cols):
        x = column_sums[col]/dc[col]
        res.append(x)

    return res

def precompute_missing(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    y = np.nan_to_num(y, nan=-100.0)
    y_pred = np.nan_to_num(y_pred, nan=-100.0)
    # Find indices where y is -100
    indices_to_change = np.where(y == -100.0)

    # Set corresponding elements in y and y_pred to 0
    y[indices_to_change] = 0.0
    y_pred[indices_to_change] = 0.0
    zeros_per_column = np.count_nonzero(y, axis=0)

    list_from_array = zeros_per_column.tolist()
    dc = {}
    for i in range(len(list_from_array)):
        dc[i] = list_from_array[i]
    return dc, y, y_pred

def evaluation_metrics(y, y_pred, eps=1e-10):
    dc, y, y_pred = precompute_missing(y, y_pred)

    mse_st = (y - y_pred) ** 2
    mae_st = np.absolute(y - y_pred)

    mse = sum_elements_per_column(mse_st, dc)
    mae = sum_elements_per_column(mae_st, dc)

    #mse = [sum(x)/len(mse_st) for x in zip(*mse_st)]
    #mae = [sum(x)/len(mae_st) for x in zip(*mae_st)]

    a = np.absolute(y - y_pred)
    b = np.absolute(y) + np.absolute(y_pred)+ eps
    norm_error_st = (a/b)

    norm_error = sum_elements_per_column(norm_error_st, dc)
    #[sum(x)*100/len(norm_error_st) for x in zip(*norm_error_st)]

    return mse, mae, norm_error




def parse_edge_list(edge_list_str):
    """
    Parse the string representation of edge list into a list of tuples.

    Parameters:
    edge_list_str (str): String representation of edge list

    Returns:
    list: List of tuples representing edges
    """
    # Convert string to actual tuples
    try:
        # Clean the string and evaluate it safely
        edges = ast.literal_eval('[' + edge_list_str + ']')
        return edges
    except:
        print(f"Error parsing edge list: {edge_list_str}")
        return []


def create_graph_from_edges(edges):
    """
    Create a NetworkX graph from a list of edges.

    Parameters:
    edges (list): List of tuples representing edges

    Returns:
    networkx.Graph: NetworkX graph object
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def visualize_graph(G, title, output_path=None):
    """
    Visualize a graph using matplotlib.

    Parameters:
    G (networkx.Graph): NetworkX graph object
    title (str): Title for the plot
    output_path (str, optional): Path to save the visualization
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color='lightblue',
                           node_size=500)

    # Draw edges
    nx.draw_networkx_edges(G, pos,
                           edge_color='gray',
                           width=1)

    # Draw labels
    nx.draw_networkx_labels(G, pos,
                            font_size=8,
                            font_weight='bold')

    plt.title(title)
    plt.axis('off')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def process_csv_file(csv_path, output_dir='graph_outputs'):
    """
    Process the CSV file and create graphs for each row.

    Parameters:
    csv_path (str): Path to the CSV file
    output_dir (str): Directory to save graph visualizations

    Returns:
    dict: Dictionary of graph objects with graph_ids as keys
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Dictionary to store graphs
    graphs = {}

    # Process each row
    for _, row in df.iterrows():
        graph_id = row['graph_id']
        edge_list = parse_edge_list(row['edge_list'])

        # Create graph
        G = create_graph_from_edges(edge_list)
        graphs[graph_id] = G

        # Save visualization
        output_file = output_path / f"{graph_id}.png"
        #visualize_graph(G, f"Graph {graph_id}", str(output_file))

    return graphs


def main():
    # Example usage
    csv_path = "output.csv"  # Replace with your CSV file path
    graphs = process_csv_file(csv_path)

    with open('ground_truth.json', 'r') as file:
        loaded_dict = json.load(file)

    MAE = 0
    # Access individual graphs
    for graph_id, G in graphs.items():
        # You can perform additional analysis on each graph here
        y_pred = calculate_stats_graph(G)
        y_ground_truth = loaded_dict[graph_id][0]
        absolute_errors = [abs(a - b) for a, b in zip(y_pred, y_ground_truth)]
        mae_st = np.mean(absolute_errors)
        MAE += mae_st

    print("MAE", MAE / (len(graphs)))

if __name__ == '__main__':
    main()