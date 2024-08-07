import osmnx as ox
import networkx as nx
import os
ox.config(use_cache=True, log_console=True)

"""Script used to generate N subgraphs from a street network city graph."""

from utg import utils
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging as log
from pathlib import Path
import pickle
import logging
logger = logging.getLogger(__name__)
import pandas as pd


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="defaults",
)

def plot_features(config: DictConfig) -> None:
    """Plot features from a set of graphs
    Args:
        config (DictConfig): Hydra config with all experiments parameters
    """

    graphs_dir = config["inspect_features"]["raw_graph_path"]
    graphs_dir = Path(graphs_dir)  

    graphs_feat = {}
    features_list = []
    for i, file in enumerate(graphs_dir.rglob("*")) :
        dict1 = {}
        label = file.stem
        if file.name.endswith(".graphml") :
            logging.info(f"Loaded file : {file.name}")
            location_name = file.parent.name
            graph = ox.load_graphml(file)
            empty_nodes = []  
            for node,attr in graph.nodes(data=True) : 
                if len(attr) == 0 : 
                    empty_nodes.append(node)
            graph.remove_nodes_from(empty_nodes)
            if config["inspect_features"]["embedding"] : 
                features = utils.generate_features(graph,config["inspect_features"]["features_to_inspect"])
            else : 
                features = utils.inspect_features(graph,config["inspect_features"]["features_to_inspect"])
            dict1 = {'label':label,'features':features}

            features_list.append(dict1)
    features_df = pd.DataFrame(features_list)
    utils.plot_features(features_df,embedding = config["inspect_features"]["embedding"],normalise=config["inspect_features"]["normalise"])

if __name__ == "__main__":
    plot_features()