import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def train_model(model, data, target, epochs=100, lr=0.001):
    logger.info(f'Starting model training for {epochs} epochs')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        out = torch.sigmoid(out)
        loss = F.binary_cross_entropy(out.view(-1, 1), target.view(-1, 1))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch + 1}: Loss = {loss.item():.4f}')
    logger.info('Training completed')

def evaluate_model(model, data, target):
    logger.info('Evaluating model')
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index)
        predictions = torch.sigmoid(predictions).numpy()
        predicted_classes = (predictions > 0.5).astype(int)
        target_np = target.numpy()
        accuracy = (predicted_classes.flatten() == target_np).mean()
        logger.info(f'Accuracy: {accuracy:.4f}')
    return accuracy
