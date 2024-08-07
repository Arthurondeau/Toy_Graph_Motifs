import osmnx as ox
import networkx as nx
import os
ox.config(use_cache=True, log_console=True)

"""Script used to generate N subgraphs from a street network city graph."""

from utg.utils import threshold_filter, convert_to_multidigraph_with_keys
import momepy as mp
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging as log
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="defaults",
)

def backbone_extractor(config: DictConfig) -> None:
    """Generate the backbone of the graph

    Args:
        config (DictConfig): Hydra config with all experiments parameters
    """

    graphs_dir = Path(config["backbone"]["graph_path"])
    save_dir = config["backbone"]["backbone_path"]
    
    if not os.path.isdir(save_dir) :
        os.mkdir(save_dir)  

    for i, file in enumerate(graphs_dir.rglob("*")) :
        save_dir_emb = config["backbone"]["backbone_path"]
        if file.name.endswith(".graphml") :
            logging.info(f"Loaded file : {file.name}")
            location_name = file.parent.name
            graph = ox.load_graphml(file)
            if (isinstance(graph,nx.MultiDiGraph)) : 
                graph = ox.convert.to_undirected(graph)
            graph = mp.straightness_centrality(ox.convert.to_digraph(graph),weight="length",name="straightness")
            backbone = threshold_filter(graph, config["backbone"]["threshold"],"straightness")
            if (isinstance(backbone,nx.DiGraph)) : 
                backbone = convert_to_multidigraph_with_keys(backbone,config['crs'])
            ox.io.save_graphml(backbone, filepath=os.path.join(save_dir, file.name + "_bc_" +  ".graphml"))
            ox.plot_graph(backbone,save=True,filepath=os.path.join(save_dir, file.name + "_bc_" +  ".png"))

if __name__ == "__main__":
    backbone_extractor()