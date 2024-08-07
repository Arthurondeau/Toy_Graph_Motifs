import osmnx as ox
import os
ox.config(use_cache=True, log_console=True)

"""Script used to create Random SubGraphs from a city query."""

from utg import create_modular_graph as cg
from utg import utils
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx 
from pathlib import Path
import shapely
import geopandas as gpd

@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="defaults",
)

def create_subgraphs(config: DictConfig) -> None:
    """Create Random SubGraphs from a city query

    Args:
        config (DictConfig): Hydra config with all experiments parameters
        datapath (Path): Path to the folder to save the subgraphs 
    """
    location_type = config["location_graph"]["type_location"]
    lat,long = config["location_graph"][location_type]["location_point"].split(",")
    dist = config["location_graph"][location_type]["dist"]
    dist_type = config["location_graph"][location_type]["dist_type"]
    network_type = config["location_graph"][location_type]["network_type"]



    G = ox.graph_from_point((float(lat),float(long)), dist=dist, dist_type=dist_type, network_type=network_type)
    save_dir = config["location_graph"]["sav_dir"]
    
    if not os.path.isdir(save_dir) :
        os.mkdir(save_dir)
    location_type_dir = os.path.join(save_dir, location_type)     
    if not os.path.isdir(location_type_dir):
        os.mkdir(location_type_dir)

    location_type_dir = os.path.join(save_dir, location_type)     
    ox.io.save_graphml(G, filepath=os.path.join(location_type_dir, location_type + ".graphml"))
    ox.plot_graph(G,save=True,filepath=os.path.join(location_type_dir, location_type + ".png"))

if __name__ == "__main__":
    create_subgraphs()