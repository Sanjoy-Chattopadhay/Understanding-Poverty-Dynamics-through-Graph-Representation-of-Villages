import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import logging
from logging_config import setup_logger
from sklearn.preprocessing import StandardScaler
from data_loader import load_village_data
from graph_creation import create_graph_with_features, add_edges_based_on_distance, calculate_betweenness_centrality, calculate_pagerank
from graph_conversion import graph_to_data
from model import GraphSAGE, train_model, evaluate_model

# Setup logging
setup_logger()

logger = logging.getLogger(__name__)

try:
    logger.info('Starting the process')
    file_path = 'village_dataset.xlsx'
    village_data = load_village_data(file_path, num_rows=150)

    Gvillage = create_graph_with_features(village_data)
    Gvillage = add_edges_based_on_distance(Gvillage, village_data, distance_threshold=10)

    # Calculate and print Betweenness Centrality
    betweenness_centrality = calculate_betweenness_centrality(Gvillage)

    # Calculate and print PageRank
    pagerank = calculate_pagerank(Gvillage)

    # Convert the graph to PyTorch Geometric format
    village_data_pyg = graph_to_data(Gvillage)
    scaler = StandardScaler()
    village_data_pyg.x = torch.tensor(scaler.fit_transform(village_data_pyg.x.numpy()), dtype=torch.float)

    target = (torch.rand(village_data_pyg.x.size(0)) > 0.5).float()

    model = GraphSAGE(in_channels=village_data_pyg.x.size(1), hidden_channels=16, out_channels=1)
    train_model(model, village_data_pyg, target)

    accuracy = evaluate_model(model, village_data_pyg, target)
    logger.info(f'Final Accuracy: {accuracy:.4f}')
except Exception as e:
    logger.error(f'Error occurred: {e}')
