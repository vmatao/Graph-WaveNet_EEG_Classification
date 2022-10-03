# Graph WaveNet for Deep Spatial-Temporal Graph Modeling

This is an adaptation to the original pytorch implementation of Graph WaveNet in the following paper: 
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121). 

Classify EEG data from the bci 2a in 5 categories: l hand, r hand, tongue, legs, no movement

<p align="center">
  <img width="350" height="400" src=./fig/model.png>
</p>

## Requirements
- python 3
- see `requirements.txt`


## Data Preparation

### Step1: Download bci 2a data from https://www.bbci.de/competition/iv/.

### Step2: Process raw data 

```
# Create data directories
mkdir -p data/{BCI}

# METR-LA


