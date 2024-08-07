"""Script used to create the template graphs found in the template_graph folder."""

from utg import create_graph as cg
from utg import utils
import hydra
from omegaconf import DictConfig
import os 
import osmnx as ox 
import networkx as nx

@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="defaults",
)

def construct_graphs(config: DictConfig) -> None:
    sav_dir = config["graph_constructor"]["sav_dir"]
    if not os.path.isdir(sav_dir):
        os.mkdir(sav_dir)
    
    graphname = "toy_concentric_smaller"
    G = cg.create_concentric_graph_old(radial=2, zones=3,straight_edges=False)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_concentric_small_with_center"
    G = cg.create_concentric_graph_old(radial=9, zones=4,straight_edges=False)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_concentric_small"
    G = cg.create_concentric_graph_old(radial=9, zones=4,straight_edges=False,center=False)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )   
    
    graphname = "toy_concentric_large_curved_with_center"
    G = cg.create_concentric_graph_old(radial=12, zones=6,straight_edges=False)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    
    graphname = "toy_concentric_large_curved"
    G = cg.create_concentric_graph_old(radial=12, zones=6,straight_edges=False,center=False)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_concentric_large_straight_with_center"
    G = cg.create_concentric_graph_old(radial=15, zones=6, straight_edges=True)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    
    graphname = "toy_concentric_large_straight"
    G = cg.create_concentric_graph_old(radial=15, zones=6, straight_edges=True,center=False)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_cross"
    G = cg.create_radial_graph()
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_star"
    G = cg.create_radial_graph(radial=8)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    
    graphname = "toy_block"
    G = cg.create_grid_graph()
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_multiple_block"
    G = cg.create_grid_graph(rows=20,cols=7,width=300,height=70)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "distorted_toy_multiple_block"
    spacing=0.95
    G = cg.create_distorted_grid_graph(rows=20,cols=7,width=300,height=70,spacing=spacing)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + str(spacing) + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + str(spacing) + ".png"
    )

    graphname = "distorted_toy_multiple_block"
    spacing=0.50
    G = cg.create_distorted_grid_graph(rows=20,cols=7,width=300,height=70,spacing=spacing)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + str(spacing) + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + str(spacing) + ".png"
    )
    graphname = "distorted_toy_multiple_block"
    spacing=0.40
    G = cg.create_distorted_grid_graph(rows=20,cols=7,width=300,height=70,spacing=spacing)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + str(spacing) + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + str(spacing) + ".png"
    )
    graphname = "distorted_toy_multiple_block"
    spacing=0.30
    G = cg.create_distorted_grid_graph(rows=20,cols=7,width=300,height=70,spacing=spacing)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + str(spacing) + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + str(spacing) + ".png"
    )
    graphname = "distorted_toy_multiple_block"
    spacing=0.20
    G = cg.create_distorted_grid_graph(rows=20,cols=7,width=300,height=70,spacing=spacing)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + str(spacing) + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + str(spacing) + ".png"
    )
    graphname = "distorted_toy_multiple_block"
    spacing=0.10
    G = cg.create_distorted_grid_graph(rows=20,cols=7,width=300,height=70,spacing=spacing)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + str(spacing) + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + str(spacing) + ".png"
    )
    graphname = "distorted_toy_multiple_block"
    spacing=0.75
    G = cg.create_distorted_grid_graph(rows=20,cols=7,width=300,height=70,spacing=spacing)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + str(spacing) + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + str(spacing) + ".png"
    )
    graphname = "toy_manhattan"
    G = cg.create_grid_graph(rows=20, cols=7, width=300, height=70, diagonal=True)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_manhattan_w_diag"
    G = cg.create_grid_graph(rows=15, cols=10, width=300, height=70, diagonal=True)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_barcelona"
    G = cg.create_grid_graph(rows=15, cols=15, diagonal=True)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    
    graphname = "toy_bridge_small"
    G = cg.create_bridge_graph()
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_bridge_large"
    G = cg.create_bridge_graph(outrows=10, sscols=10, bridges=2)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_fractal_cross"
    G = cg.create_fractal_graph(branch=4, level=2)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_fractaler_cross"
    G = cg.create_fractal_graph(branch=4, level=3)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_fractalerer_cross"
    G = cg.create_fractal_graph(branch=4, level=4)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    """
    graphname = "toy_balanced_tree"
    G = nx.balanced_tree(r=3, h=5, create_using=None)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_cycle_graph"
    G = nx.cycle_graph(r=100, create_using=None)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_lollilop_graph"
    G = nx.lollipop_graph(m=30,n=15, create_using=None)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_tadpole_graph"
    G = nx.tadpole_graph(m=30,n=15, create_using=None)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    graphname = "toy_wheel_graph"
    G = nx.wheel_graph(n=20, create_using=None)
    G.graph["crs"] = config["crs"] 
    utils.save_graph(G, sav_dir + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath=sav_dir + graphname + ".png"
    )
    """
if __name__ == "__main__":
    construct_graphs()

