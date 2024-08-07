import osmnx as ox
import networkx as nx
import os
ox.config(use_cache=True, log_console=True)

"""Script used to plot the UMAP projection of graphs embeddings with associated clusters."""

from utg import utils
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging as log
from pathlib import Path
import pickle
import pandas as pd


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="defaults",
)

def motifs_clustering(config: DictConfig) -> None:
    """Run UMAP Projection + HDBSCAN Clustering

    Args:
        config (DictConfig): Hydra config with all experiments parameters
    """

    graphs_dir_training = config["clustering"]["graph_path_training"]    
    graphs_dir_training = Path(graphs_dir_training)  
    graphs_dir_predict = config["clustering"]["graph_path_predict"]    
    graphs_dir_predict = Path(graphs_dir_predict)      
    training_df = utils.generate_df_emb(graphs_dir_training)
    predict_df = utils.generate_df_emb(graphs_dir_predict)


    
    find_clusters = utils.plot_clusters(training_df,
                                        predict_df,
                                        normalise = config["clustering"]["normalise"],
                                        features_selection=config["clustering"]["features_selection"],
                                        features_list=config["embedding"]["detailed_features_list"],
                                        compare_cluster_dist=config["clustering"]["compare_cluster_dist"],
                                        Kbest_feats=config["clustering"]["Kbest_feats"],
                                        n_neighbors= config["clustering"]["n_neighbors"],
                                        min_dist=config["clustering"]["min_dist"],
                                        n_components=config["clustering"]["n_components"],
                                        random_state=config["clustering"]["random_state"],
                                        metric=config["clustering"]["metric"],
                                        )
if __name__ == "__main__":
    motifs_clustering()