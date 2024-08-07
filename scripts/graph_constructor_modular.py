"""Script used to create the template graphs found in the template_graph folder."""

from utg import create_modular_graph as cg
from utg import utils
import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="defaults",
)

def create_modular_graph(config: DictConfig) -> None:
    """Create Modular Graph

    Args:
        config (DictConfig): Hydra config with all experiments parameters
        datapath (Path): Path to the folder with training graphs
    """
    graphname = config["modular_graph"]["graph_name"]
    modular_module = hydra.utils.instantiate(config["modular_graph"]["create_modular_graph"])
    G = modular_module.create_modular_graph()
    
    utils.save_graph(G, "./template_graph/" + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath="./template_graph/" + graphname + ".png"
    )

if __name__ == "__main__":
    create_modular_graph()