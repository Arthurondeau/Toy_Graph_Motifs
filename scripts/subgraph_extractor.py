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
    city_name = config["subgraphs"]["subgraph_creation"]["city_name"]

    # City Query
    query = {'city': city_name}

    # get the boundaries of the place
    gdf = ox.geocode_to_gdf(query)
    area = ox.projection.project_gdf(gdf).unary_union.area
    #gdf.plot()

    # get the street network within the city
    G = ox.graph_from_place(query, network_type='drive')

    save_dir = config["subgraphs"]["city_graph"]["save_path"]
    
    if os.path.isdir(save_dir) : 
        city_dir = os.path.join(save_dir, city_name)
        if not os.path.isdir(city_dir):
            os.mkdir(city_dir)        
        ox.io.save_graphml(G, filepath=os.path.join(city_dir, city_name + ".graphml"))
    else : 
        os.mkdir(save_dir)
        city_dir = os.path.join(save_dir, city_name)
        if not os.path.isdir(city_dir):
            os.mkdir(city_dir)   
        ox.io.save_graphml(G, filepath=os.path.join(city_dir, city_name + ".graphml"))

    samples_indices = [] #Stored indices of all generated subgraphs
    subgraph_dir = config["subgraphs"]["subgraph_creation"]["save_path"]
    if not os.path.isdir(subgraph_dir):
        os.mkdir(subgraph_dir)
    subgraph_city_dir = os.path.join(subgraph_dir,city_name)
    if not os.path.isdir(subgraph_city_dir):
        os.mkdir(subgraph_city_dir)

    for i in tqdm(range(config["subgraphs"]["subgraph_creation"]["num_subgraphs_per_graph"])):
        subgraph, samples_indices,sampled_idx_subgraph = utils.sample_subgraphs_from_graph(
                    G,
                    samples_indices,
                    d_to_centroid=config["subgraphs"]["subgraph_creation"]["d_to_centroid"],
                    overlap_threshold=config["subgraphs"]["subgraph_creation"]["overlap_threshold"],
        )
        ox.save_graphml(subgraph,Path(os.path.join(subgraph_city_dir, city_name + str(i) + ".graphml")))
        #Plot graph and latest subgraph
        sav_path = Path(os.path.join(subgraph_city_dir, city_name + str(i) + ".png"))
        utils.plot_subgraph(G,sampled_idx_subgraph,sav_path)
     


    

if __name__ == "__main__":
    create_subgraphs()