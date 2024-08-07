import osmnx as ox
import networkx as nx
import os

ox.config(use_cache=True, log_console=True)

"""Script used to compute the toyness score for a set of graphs."""

from utg import utils
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

def graph_toyness_score(config: DictConfig) -> None:
    """Compute toyness score from a graphs set

    Args:
        config (DictConfig): Hydra config with all experiments parameters
    """

    graphs_dir = config["toyness_score"]["graphs_path"]
    graphs_dir = Path(graphs_dir)  

    graphs_dict = {}
    metrics = config["toyness_score"]["metrics"]
    for i, file in enumerate(graphs_dir.rglob("*")) :
        if file.name.endswith(".graphml") :
            logging.info(f"Loaded file : {file.name}")
            location_name = file.name
            graph = ox.load_graphml(file)
            if (isinstance(graph,nx.MultiDiGraph)) : 
                graph = ox.convert.to_undirected(graph)
            empty_nodes = [] 
            if not "toy" or not "bc" in location_name : 
                for node,attr in graph.nodes(data=True) : 
                    if len(attr) == 0 : 
                        empty_nodes.append(node)
                graph.remove_nodes_from(empty_nodes)

            graphs_dict[location_name] = graph   


    graphs_scores = utils.toyness_scores_metrics(graphs_dict,metrics,threshold=0.43)
    #utils.plot_length(graphs_dict)

if __name__ == "__main__":
    graph_toyness_score()