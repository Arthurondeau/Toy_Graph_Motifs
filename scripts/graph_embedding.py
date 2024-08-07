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


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="defaults",
)

def graph_embedding(config: DictConfig) -> None:
    """Generate graph embedding

    Args:
        config (DictConfig): Hydra config with all experiments parameters
    """

    graphs_dir = config["embedding"]["raw_graph_path"]
    save_dir_emb = config["embedding"]["sav_dir"]
    if not os.path.isdir(save_dir_emb) : 
        os.mkdir(save_dir_emb)
    graphs_dir = Path(graphs_dir)  

    for i, file in enumerate(graphs_dir.rglob("*")) :
        save_dir_emb = config["embedding"]["sav_dir"]
        if file.name.endswith(".graphml") :
            logging.info(f"Loaded file : {file.name}")
            location_name = file.parent.name
            graph = ox.load_graphml(file)
            if (isinstance(graph,nx.MultiDiGraph)) : 
                graph = ox.convert.to_undirected(graph)
            if utils.number_connected_components(graph) != 1 : 
                print('Uncorrect graph:',location_name)

            empty_nodes = []  
            for node,attr in graph.nodes(data=True) : 
                if len(attr) == 0 : 
                    empty_nodes.append(node)
            graph.remove_nodes_from(empty_nodes)

            features = utils.generate_features(graph,config["embedding"]["features_list"])
            save_dir_emb = os.path.join(save_dir_emb,location_name)
            if not os.path.isdir(save_dir_emb):
                os.mkdir(save_dir_emb)
            with open(os.path.join(save_dir_emb, "emb_" + file.stem + ".pck"), 'wb') as handle:
                pickle.dump(features,handle)
            log.info("Generated embedded graph","emb_" + file.stem + ".pck" )     

if __name__ == "__main__":
    graph_embedding()