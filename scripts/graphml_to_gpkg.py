import osmnx as ox
import networkx as nx
import os
ox.config(use_cache=True, log_console=True)

from utg import utils
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging as log
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="defaults",
)

def graphml_to_gpkg(config: DictConfig) -> None:
    """Generate the graphml file from gpkg

    Args:
        config (DictConfig): Hydra config with all experiments parameters
    """

    graphs_dir = config["clean_graph"]["raw_graph_path"]
    save_dir_emb = config["clean_graph"]["sav_dir_raw"]
    if not os.path.isdir(save_dir_emb) : 
        os.mkdir(save_dir_emb)
    graphs_dir = Path(graphs_dir)  

    for i, file in enumerate(graphs_dir.rglob("*")) :
        save_dir_emb = config["clean_graph"]["sav_dir_raw"]
        if file.name.endswith(".graphml") :
            logging.info(f"Loaded file : {file.name}")
            location_name = file.parent.name
            graph = ox.load_graphml(file)
            save_dir_emb = os.path.join(save_dir_emb,location_name)
            if not os.path.isdir(save_dir_emb):
                os.mkdir(save_dir_emb)
            ox.io.save_graph_geopackage(graph, filepath=os.path.join(save_dir_emb, file.stem + ".gpkg"))
            log.info("Generated cleaned graph",file.stem + ".gpkg" )     

if __name__ == "__main__":
    graphml_to_gpkg()

