# Understanding-Poverty-Dynamics-through-Graph-Representation-of-Villages

# Village-Level Poverty Assessment using Graph Neural Networks

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Graph Structure](#graph-structure)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## Introduction
This project aims to assess village-level poverty using graph-based models that incorporate geographical and socio-economic features. The main objective is to identify influential villages based on connectivity and centrality measures, leveraging Graph Neural Networks (GNNs) for accurate predictions.

## Features
- Graph construction from village data
- Calculation of Betweenness Centrality and PageRank
- Implementation of the GraphSAGE model for poverty prediction
- Logging and error handling for efficient debugging

## Graph Structure
- The graph consists of nodes representing villages.
- Edges are established based on geographic proximity.
- Each node includes the following features:
  - Population
  - Latitude
  - Longitude
  - Availability of essential services (e.g., schools, healthcare, transportation)

## Model Architecture
- The model utilizes a GraphSAGE architecture, which consists of:
  - **Input Layer:** Features from each village
  - **Hidden Layer:** 16 units
  - **Output Layer:** Binary classification for poverty status
- Centrality measures, including:
  - PageRank
  - Betweenness Centrality
- These measures are integrated into the model to enhance predictions.

## Results
- The model achieves an accuracy of approximately 80% in predicting poverty levels across different villages.
- Detailed results and metrics can be found in the logging output.
