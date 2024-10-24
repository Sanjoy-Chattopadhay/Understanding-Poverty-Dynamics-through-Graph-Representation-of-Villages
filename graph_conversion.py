import logging
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

def graph_to_data(G):
    logger.info('Converting graph to PyTorch Geometric format')
    try:
        edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
        node_map = {old_id: new_id for new_id, old_id in enumerate(G.nodes())}
        edge_index = torch.tensor([[node_map[edge_index[0][i].item()] for i in range(edge_index.size(1))],
                                   [node_map[edge_index[1][i].item()] for i in range(edge_index.size(1))]], 
                                   dtype=torch.long)
        node_features = []
        for node_id, features in G.nodes(data=True):
            feature_vector = []
            for key in sorted(features.keys()):
                value = features[key]
                if isinstance(value, (int, float)):
                    feature_vector.append(value)
                else:
                    feature_vector.append(0.0)
            node_features.append(feature_vector)
        node_features = torch.tensor(node_features, dtype=torch.float)
        data = Data(x=node_features, edge_index=edge_index)
        logger.info('Graph successfully converted to PyTorch Geometric format')
        return data
    except Exception as e:
        logger.error(f'Error during graph conversion: {e}')
        raise
