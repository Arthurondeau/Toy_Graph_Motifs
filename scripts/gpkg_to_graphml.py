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
import geopandas as gpd
import momepy

@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="defaults",
)



def gpkg_to_graphml(config: DictConfig) -> None:

    """Generate graphml file from a gpkg file 

    Args:
        config (DictConfig): Hydra config with all parameters
    """


    graphs_dir = config["clean_graph"]["sav_dir_clean"]
    save_dir_emb = config["clean_graph"]["sav_dir_clean"]
    if not os.path.isdir(save_dir_emb) : 
        os.mkdir(save_dir_emb)
    graphs_dir = Path(graphs_dir) 
  
    for i, file in enumerate(graphs_dir.rglob("*")) :
        save_dir_emb = config["clean_graph"]["sav_dir_clean"]
        if file.name.endswith(".gpkg") :
            logging.info(f"Loaded file : {file.name}")
            location_name = file.parent.name
            gdf_nodes = gpd.read_file(file, layer='nodes').set_index('osmid')
            gdf_edges = gpd.read_file(file, layer='edges').set_index(['u', 'v', 'key'])
            assert gdf_nodes.index.is_unique and gdf_edges.index.is_unique
            graph_attrs = {'crs': 'epsg:4326', 'simplified': True}
            graph = ox.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs)
            empty_nodes = [] 
            if not "toy" or not "bc" in location_name : 
                for node,attr in graph.nodes(data=True) : 
                    if len(attr) == 0 : #remove remaining empty nodes after cleaning with QGIS
                        empty_nodes.append(node)
                graph.remove_nodes_from(empty_nodes)
            save_dir_emb = os.path.join(save_dir_emb,location_name)

            if not os.path.isdir(save_dir_emb):
                os.mkdir(save_dir_emb)

            print('path : ',os.path.join(save_dir_emb, file.stem + ".graphml"))
            utils.save_graph(graph, os.path.join(save_dir_emb, file.stem + ".graphml"))
            ox.plot_graph(graph,save=True,filepath=os.path.join(save_dir_emb, file.stem + ".png"))
            log.info("Generated cleaned graph",file.stem + ".graphml" ) 
if __name__ == "__main__":
    gpkg_to_graphml()

