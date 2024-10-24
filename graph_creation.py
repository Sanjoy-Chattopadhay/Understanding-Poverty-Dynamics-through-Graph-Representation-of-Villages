import logging
import networkx as nx
from geopy.distance import geodesic

logger = logging.getLogger(__name__)

def create_graph_with_features(village_data):
    logger.info('Creating graph with village features')
    G = nx.Graph()
    for _, row in village_data.iterrows():
        village_id = int(row['Village_ID'])
        features = {
            'Population': row['Population'],
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'Senior_Secondary_School': row['Senior_Secondary_School'],
            'College': row['College'],
            'Primary_Health_Sub_Centre': row['Primary_Health_Sub_Centre'],
            'Tap_Drinking_Water': row['Tap_Drinking_Water'],
            'Bus_Facility': row['Bus_Facility'],
            'Approach_by_pucca_road': row['Approach_by_pucca_road'],
            'Banks': row['Banks']
        }
        G.add_node(village_id, **features)
    logger.info(f'Graph created with {G.number_of_nodes()} nodes')
    return G

def add_edges_based_on_distance(G, village_data, distance_threshold=10):
    logger.info(f'Adding edges between villages within {distance_threshold} km distance')
    nodes = list(G.nodes(data=True))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_i, attrs_i = nodes[i]
            node_j, attrs_j = nodes[j]
            coords_i = (attrs_i['Latitude'], attrs_i['Longitude'])
            coords_j = (attrs_j['Latitude'], attrs_j['Longitude'])
            distance = geodesic(coords_i, coords_j).kilometers
            if distance <= distance_threshold:
                G.add_edge(node_i, node_j, weight=distance)
    logger.info(f'Edges added. Total edges: {G.number_of_edges()}')
    return G

def calculate_betweenness_centrality(G):
    logger.info('Calculating Betweenness Centrality')
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    logger.info('Betweenness Centrality calculated')
    
    # Print betweenness centrality for each node
    for node, centrality in betweenness_centrality.items():
        print(f'Village ID: {node}, Betweenness Centrality: {centrality:.4f}')
        logger.debug(f'Village ID: {node}, Betweenness Centrality: {centrality:.4f}')
    
    return betweenness_centrality

def calculate_pagerank(G, alpha=0.85):
    logger.info('Calculating PageRank')
    pagerank = nx.pagerank(G, alpha=alpha)
    logger.info('PageRank calculated')
    
    # Print PageRank for each node
    for node, rank in pagerank.items():
        print(f'Village ID: {node}, PageRank: {rank:.4f}')
        logger.debug(f'Village ID: {node}, PageRank: {rank:.4f}')
    
    return pagerank
