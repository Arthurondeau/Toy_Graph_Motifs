"""Useful functions."""

import logging as log
import math
import os
import pickle
from pathlib import Path
from typing import List, Optional, Set, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import momepy as mp
import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
import shapely.ops
from umap import UMAP
import hdbscan
import diptest

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score, 
                             pairwise_distances, precision_score, recall_score, 
                             roc_curve)
from sklearn.metrics.pairwise import manhattan_distances

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy
from scipy import optimize, spatial as sp, stats
from scipy.stats import (entropy, kurtosis, rankdata, shapiro, skew, tstd, zscore, kstest)
from shapely.geometry import (LineString, MultiLineString, MultiPoint, 
                              Point, Polygon)
from shapely.ops import unary_union, polygonize
from shapely import union,union_all
from mpl_toolkits.mplot3d import Axes3D
import igraph as ig


def make_osmnx_compatible(G):
    """Make the graph osmnx-compatible."""
    G = G.copy()
    for c, edge in enumerate(G.edges):
        G.edges[edge]["osmid"] = c
    G.graph["crs"] = "epsg:2154"
    G.graph["simplified"] = True
    if type(G) != nx.MultiDiGraph:
        G = nx.MultiDiGraph(G)
    return G


def save_graph(G, filepath=None, gephi=False, encoding="utf-8"):
    """
    Save graph to disk as GraphML file.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    filepath : string or pathlib.Path
        path to the GraphML file including extension. if None, use default
        data folder + graph.graphml
    gephi : bool
        if True, give each edge a unique key/id to work around Gephi's
        interpretation of the GraphML specification
    encoding : string
        the character encoding for the saved file

    Returns
    -------
    None
    """
    filepath = Path(filepath)
    # if save folder does not already exist, create it
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if gephi:
        # for gephi compatibility, each edge's key must be unique as an id
        uvkd = ((u, v, k, d) for k, (u, v, d) in enumerate(G.edges(keys=False, data=True)))
        G = nx.MultiDiGraph(uvkd)

    else:
        # make a copy to not mutate original graph object caller passed in
        G = G.copy()

    # stringify all the graph attribute values
    for attr, value in G.graph.items():
        G.graph[attr] = str(value)

    # stringify all the node attribute values
    for _, data in G.nodes(data=True):
        for attr, value in data.items():
            data[attr] = str(value)

    # stringify all the edge attribute values
    for _, _, data in G.edges(data=True):
        for attr, value in data.items():
            data[attr] = str(value)

    nx.write_graphml(G, path=filepath, encoding=encoding)
    print('Saved as Graphml graph :',filepath.stem)

def save_graph_unsorted(G, filepath):
    """Save the graph in the corresponding filepath, converting geometry to WKT string."""
    G = G.copy()
    for e in G.edges():
        print('e',e)
        if len(e) == 2:
            u, v = e
            print('items edges', G.get_edge_data(u, v).items())
            if "geometry" in G.edges[(u, v)].keys():
                G.edges[(u, v)]["geometry"] = shapely.to_wkt(G.edges[(u, v)]["geometry"])
        else :
            for k in range(len(e)):
                if "geometry" in G.edges[(u, v, k)].keys():
                    u, v = e
                    G.edges[(u, v, k)]["geometry"] = shapely.to_wkt(G.edges[(u, v, k)]["geometry"])
    nx.write_graphml(G, filepath)

def load_graph(filepath):
    """Load the graph from the corresponding filepath, creating geometry from WKT string."""
    G = nx.read_graphml(filepath)
    G = nx.relabel_nodes(G, lambda x: int(x))
    for e in G.edges:
        G.edges[e]["geometry"] = shapely.from_wkt(G.edges[e]["geometry"])
    return G


def plot_graph(
    G,
    square_bb=True,
    show_voronoi=False,
    show=True,
    save=False,
    close=False,
    filepath=None,
    rel_buff=0.1,
):
    """Plot the graph using geopandas plotting function, with the option to save the picture and see the Voronoi cells.

    Args:
        G (nx.Graph or nx.MultiDiGraph): Graph we want to plot.
        square_bb (bool, optional): If True, limits of the figure are a square centered around the graph. Defaults to True.
        show_voronoi (bool, optional): If True, show the Voronoi cells for each nodes. Defaults to False.
        show (bool, optional): If True, show the figure in Python. Defaults to True.
        save (bool, optional): If True, save the figure at the designated filepath. Defaults to False.
        filepath (_type_, optional): Path for the saved figure. Defaults to None.
        rel_buff (float, optional): Relative buffer around the nodes, creating padding for the square bounding box. For instance a padding of 10% around each side of the graph for a value of 0.1. Defaults to 0.1.

    Raises:
        ValueError: If save is True, need to specify a filepath. Filepath can't be None.
    """
    fig, ax = plt.subplots()
    geom_node = [shapely.Point(get_node_coord(G, node)) for node in G.nodes]
    geom_edge = list(nx.get_edge_attributes(G, "geometry").values())
    gdf_node = gpd.GeoDataFrame(geometry=geom_node)
    gdf_edge = gpd.GeoDataFrame(geometry=geom_edge)
    gdf_edge.plot(ax=ax, color="steelblue", zorder=1, linewidth=2)
    gdf_node.plot(ax=ax, color="black", zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    bounds = gdf_node.total_bounds
    if square_bb:
        # Find if graph is larger in width or height
        side_length = max(bounds[3] - bounds[1], bounds[2] - bounds[0])
        # Find center of the graph
        mean_x = (bounds[0] + bounds[2]) / 2
        mean_y = (bounds[1] + bounds[3]) / 2
        # Add padding
        xmin = mean_x - (1 + rel_buff) * side_length / 2
        xmax = mean_x + (1 + rel_buff) * side_length / 2
        ymin = mean_y - (1 + rel_buff) * side_length / 2
        ymax = mean_y + (1 + rel_buff) * side_length / 2
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    if show_voronoi:
        # Create a bounding box to create bounded Voronoi cells that can easily be drawn
        vor_buff = max(xmax - xmin, ymax - ymin)
        bb = np.array(
            [xmin - vor_buff, xmax + vor_buff, ymin - vor_buff, ymax + vor_buff]
        )
        bounded_vor = bounded_voronoi([get_node_coord(G, node) for node in G.nodes], bb)
        vor_cells = create_voronoi_polygons(bounded_vor)
        gdf_voronoi = gpd.GeoDataFrame(geometry=vor_cells)
        gdf_voronoi.geometry = gdf_voronoi.geometry.exterior
        gdf_voronoi.plot(ax=ax, color="firebrick", alpha=0.7, zorder=0)
    if show:
        plt.show()
    if save:
        if filepath is None:
            raise ValueError("If save is True, need to specify a filepath")
        fig.savefig(filepath, dpi=300)
    if close:
        plt.close()
    return fig, ax


def make_true_zero(vec):
    """Round to zero when values are very close to zero in a list."""
    return [round(val) if math.isclose(val, 0, abs_tol=1e-10) else val for val in vec]


def get_node_coord(G:any, n:int) -> list[int]:
    """Return the coordinates of the node."""
    return [G.nodes[n]["x"], G.nodes[n]["y"]]

def get_list_nodes_coords(G:any,idx:int) -> list[int]:
    """Return the coordinates list of the corresponding nodes index

    Args:
        G (any): nx.Graph()
        idx (int): Number of the graph node

    Returns:
        list[int]: Coordinates of the node
    """    """"""
    return [get_node_coord(G,i) for i in idx]

def get_knn(data_coords:list[float],query_coord:list[float],knn:int)->list[float]:
    """Return the indices of the knn of query point

    Args:
        data_coords (list[float]): list of large grid coordinates
        query_coord (list[float]): coordinates of the the query node
        knn (int): number of nearest neighbors to return

    Returns:
        list[float]: index of knn
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data_coords)
    distances, indices = nbrs.kneighbors(np.reshape(query_coord,(1,-1)))
    return indices[0]

def get_cell_nodes(cell_coor:list,rows:int) -> list:
    """Return the indices of the four cell nodes

    Args:
        cell_coor (str): coordinates of the cell
        rows (int): rows number of the grid

    Returns:
        list: node number of the cell nodes
    """    """"""
    cell_x,cell_y = cell_coor[0],cell_coor[1]
    cell_node = int(cell_x)*rows + int(cell_y)
    return [cell_node,cell_node+1,cell_node+rows,cell_node+rows+1]


def get_node_number(node_coord: list,rows_grid) -> int:
    return int(node_coord[0])*rows_grid + int(node_coord[1])


def normalize(vec):
    """Normalize the vector."""
    return vec / np.linalg.norm(vec)

def z_score(vec): 
    """ Compute the z-score"""
    vec = (vec-np.mean(vec))/np.std(vec)
    return vec 

def merge_graph(G:any,Hlist:list[any]) -> any:
    """Delete the nodes of graph H in commun with graph G

    Args:
        G (any): Input graph
        H (list[any]): List of graphs to merge on input graph

    Returns:
        any: merged graph
    """
    for H in Hlist : 
        for i in G.nodes():
            for j in list(H.nodes()):
                if H.nodes[j]["x"] == G.nodes[i]["x"] and H.nodes[j]["y"] == G.nodes[i]["y"]:
                    H.remove_node(j)
        G = nx.compose(G,H)
        G = nx.convert_node_labels_to_integers(G,ordering='sorted')
    return G


def check_is_on_edge(G:any,coord:list[float]) -> tuple:
    """Check if the query coord node is on an edge (straight line for now) 
    of the input graph

    Args:
        G (nx.Graph()): Input graph
        coord (list[float]): Query node

    Returns:
        Tuple(bool,float,float): Return True if node in on an edge of graph G, False if not
                                 and the index of edge's nodes

    """    
    for edge in G.edges():
        node1, node2 = edge
        coord1, coord2 = get_node_coord(G,node1),get_node_coord(G,node2)
        distance1, distance2 = np.linalg.norm(coord1-coord),np.linalg.norm(coord2-coord)
        distance12 = np.linalg.norm(coord1-coord2)
        if (distance12 == distance1 + distance2):
            return True, node1, node2
        else:
            return False, node1, node2

def find_angle(vec):
    """Find the angle of the vector to the origin and the horizontal axis."""
    vec = np.array(vec[1]) - np.array(vec[0])
    normvec = make_true_zero(normalize(vec))
    if normvec[1] >= 0:
        return np.arccos(normvec[0])
    elif normvec[0] >= 0:
        angle = np.arcsin(normvec[1])
        if angle < 0:
            angle += 2 * np.pi
        return angle
    else:
        return np.arccos(normvec[0]) + np.pi / 2


# TODO: Look at shapely voronoi to maybe make a change for better written code
def bounded_voronoi(points, bb):
    """Make bounded voronoi cells for points by creating a large square of artifical points far away."""
    artificial_points = []
    # Make artifical points outside of the bounding box
    artificial_points.append([bb[0], bb[2]])
    artificial_points.append([bb[0], bb[3]])
    artificial_points.append([bb[1], bb[2]])
    artificial_points.append([bb[1], bb[3]])
    for x in np.linspace(bb[0], bb[1], num=100, endpoint=False)[1:]:
        artificial_points.append([x, bb[2]])
        artificial_points.append([x, bb[3]])
    for y in np.linspace(bb[2], bb[3], num=100, endpoint=False)[1:]:
        artificial_points.append([bb[0], y])
        artificial_points.append([bb[1], y])
    points = np.concatenate([points, artificial_points])
    # Find Voronoi regions
    vor = sp.Voronoi(points)
    regions = []
    points_ordered = []
    # Keep regions for points that are within the bounding box so only the original points
    for c, region in enumerate(vor.regions):
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not (bb[0] <= x and x <= bb[1] and bb[2] <= y and y <= bb[3]):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
            points_ordered.append(np.where(vor.point_region == c)[0][0])
    # Create filtered attributes to keep in memory the original points and related Voronoi regions
    vor.filtered_points = points_ordered
    vor.filtered_regions = regions
    return vor


def create_voronoi_polygons(vor, filtered=True):
    """Create polygons from Voronoi regions. Use the filtered attributes from bounded_voronoi."""
    vor_poly = []
    attr = vor.regions
    if filtered:
        attr = vor.filtered_regions
    for region in attr:
        vertices = vor.vertices[region, :]
        vor_poly.append(shapely.Polygon(vertices))
    return vor_poly



def sample_subgraphs_from_graph(
    G: nx.Graph,
    samples_indices: List[Set[int]],
    d_to_centroid: int = 500,
    overlap_threshold: float = 0,
) -> Tuple[Optional[np.array], List[Set[int]]]:
    """randomly sample a subgraph from graph

    a random point is selected as seed,
    all points within d_to_centroid to this seed are selected
    a graph is created with those points

    if the selected nodes are have >overlap_threshold nodes in common
    with another sampled graph in sampled_indices, this graph won't be kept

    overlap_threshold=1 => all graphs are kept (can't have >1 ratio of common nodes)
    overlap_threshold=0 => 0 nodes in common

    Args:
        graph (nx.Graph): Graph with cells features and coordinates
        samples_indices (List[List[int]]): list of previously sampled nodes
        d_to_centroid (int, optional): max dist with seed. Defaults to 500.
        overlap_threshold (float, optional): 0<threshold<1. Defaults to 0.

    Returns:
    Tuple[torch.Tensor, List[Set[int]]]: adjacency matrix, updated samples_indices, subgraph_indices
    """
    cell_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    seed = np.random.randint(0, cell_df.shape[0] - 1)
    x_id, y_id = cell_df.iloc[seed]["x"], cell_df.iloc[seed]["y"]
    seed_coords = np.array([x_id, y_id])

    sampled_idx = get_nodes_in_radius(G,seed_coords,d_to_centroid)
    # check if sampled set of nodes overlap with another sampled graph
    new_adj, new_feats = None, None
    if not samples_indices:
        subG = G.subgraph(sampled_idx)
    else : 
        if not any(
                    len(sampled_idx & set(idx)) / len(sampled_idx) > overlap_threshold
                    for idx in samples_indices
                ) :
                    samples_indices += [sampled_idx]
                    subG = G.subgraph(sampled_idx)

    return subG, samples_indices,sampled_idx


def get_nodes_in_radius(G:nx.Graph,node_coord: list[int],radius:float) -> list[int] : 
    """Get the nodes located in the circle centered of a point

    Args:
        G (nx.Graph): Input graph
        node_coord (list[int]): 
        radius(float): radius of the circle (in meters)

    Returns:
        list[int]: List of the extracted nodes indices
        nx.Graph : Extracted subgraph
    """

    # create graph and extract node geometries
    gdf_nodes = ox.graph_to_gdfs(G, edges=False)[['geometry']]

    # create a GeoSeries for the single point P
    gdf_point = gpd.GeoDataFrame({'points': [Point(node_coord[0], node_coord[1])]}, geometry='points',crs='epsg:4326')

    # create buffer around the single point P
    gdf_point_meter = gdf_point.to_crs(3857) #Project into meters coordinates
    buffer = ox.project_gdf(gdf_point_meter.buffer(radius), to_latlong=True)
    gdf_buffers = gpd.GeoDataFrame(geometry=buffer)

    # find all the nodes within the buffer of the single point P
    result = gpd.sjoin(gdf_buffers, gdf_nodes, how='inner', op='intersects')
    return result['index_right'].tolist()


def plot_subgraph(G:nx.Graph,sampled_index:list[int],sav_path:Path) -> None :
    """Plot a G with its subgraph 

    Args:
        G (nx.Graph): Input graph
        sampled_index (list[int]): index of subgraph nodes
        sav_path (Path): Path of the saved graph png

    Returns:
        None
    """
    for node in G.nodes(data=True):
        print('node',node)
        if node[0] in sampled_index:
            node[1]["color"] = 'r'
        else :
            node[1]["color"] = 'b'
    # Plot the graph with nodes colored based on their attribute "color"
    ox.plot_graph(G,save=True, node_color=[G.nodes[node]['color'] for node in G.nodes()],filepath=sav_path,close=True,show=False)

def plot_node_degree(G:nx.Graph,degree:int,sav_path:Path) -> None :
    """Plot a G with node with specified degree colored

    Args:
        G (nx.Graph): Input graph
        degree (int): node degree to plot
        sav_path (Path): Path of the saved png
    Returns:
        None
    """
    colors = []
    nx.set_node_attributes(G,colors,"color")
    for node,node_degree in G.degree():
        if node_degree == degree :
            G.nodes[node]["color"] = "r"
        else :
            G.nodes[node]["color"] = "b"

    # Plot the graph with nodes colored based on their attribute "color"
    ox.plot_graph(G,save=True, node_color=[G.nodes[node]['color'] for node in G.nodes()],filepath=sav_path,close=True,show=False)


def plot_neighbors(G:nx.Graph,source_node_index: int ,index_neighbors:list[int]) -> None :
    """Plot a G with a source node and its neighbors 

    Args:
        G (nx.Graph): Input graph
        source_node_index : index of source node
        index_neighbors (list[int]): index of neighbors nodes

    Returns:
        None
    """
    for node in G.nodes(data=True):
        if node[0] in index_neighbors:
            node[1]["color"] = 'b'
        elif node[0] == source_node_index:
            node[1]["color"] = 'r'
        else :
            node[1]["color"] = 'black'
    # Plot the graph with nodes colored based on their attribute "color"
    ox.plot_graph(G, node_color=[G.nodes[node]['color'] for node in G.nodes()])
    plt.show()


def read_graphs_from_folder(folder_path):
    """return the list of graphs located in the folder

    Args:
        folder_path (str): path to graph folder

    Returns:
        list[nx.Graph]: list of the graphs
    """    
    graphs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".graphml"):
            file_path = os.path.join(folder_path, file_name)
            try:
                graph = nx.read_graphml(file_path)
                graphs.append(graph)
                print(f"Read graph from {file_name}")
            except Exception as e:
                print(f"Error reading graph from {file_name}: {e}")
    return graphs


def degree_ratio(graph:nx.Graph,degree:int) -> float:
    """Return the degree ratio in the graph

    Args:
        graph (nx.Graph): Undirected graph
        degree (int): degree

    Returns:
        float: degree ratio

    """ 
    Nb_nodes = len(graph.nodes())
    node_degrees = [node_degree for node,node_degree in graph.degree()]
    return node_degrees.count(degree)/Nb_nodes
    

def generate_features(graph:nx.Graph,features_list:list[str])->nx.Graph:
    """Generate the new features for each node based on features list

    Args:
        G (nx.Graph): Input G
        features_list (list[str]): list of metrics to compute

    Returns:
        nx.Graph: embedded graph
    """

    features = {}

    if "betweenness" in features_list:
        offset = 1e-5
        BC_distr = betweenness_centrality(graph)
        BC_kurto = kurtosis(BC_distr,fisher=False)
        BC_skew = skew(BC_distr)
        BC_IQM = interquartile_mean(BC_distr)
        features["BC_kurto"] = BC_kurto
        features["BC_skew"] = BC_skew
        features["BC_IQR"] = scipy.stats.iqr(BC_distr)
        features["BC_IQM"] = BC_IQM
    if "straightness" in features_list:

        SC_distr  = straightness_centrality(graph)
        SC_kurto = kurtosis(SC_distr,fisher=False)
        SC_skew = skew(SC_distr)
        SC_IQM = interquartile_mean(SC_distr)
        features["SC_IQR"] = scipy.stats.iqr(SC_distr)
        features["SC_kurto"] = SC_kurto
        features["SC_skew"] = SC_skew
        features["SC_IQM"] = SC_IQM

    if "circular_footprint" in features_list:

        footprint = circular_footprint(graph)

        if isinstance(footprint,np.ndarray): 
            footprint_kurto = kurtosis(footprint,fisher=False)
            footprint_skew = skew(footprint)
            footprint_IQR = scipy.stats.iqr(footprint)
            footprint_IQM = interquartile_mean(footprint)
        else : 
            footprint_kurto = None
            footprint_skew = None
            footprint_IQR = None
            footprint_IQM = None
        features["circular_footprint_kurto"] = footprint_kurto
        features["circular_footprint_skew"] = footprint_skew
        features["circular_footprint_IQR"] = footprint_IQR
        features["circular_footprint_IQM"] = footprint_IQM

    if "rec_footprint" in features_list:

        footprint = rec_footprint(graph)
        if isinstance(footprint,np.ndarray):
            footprint_kurto = kurtosis(footprint,fisher=False)
            footprint_skew = skew(footprint)
            footprint_IQR = scipy.stats.iqr(footprint)
            footprint_IQM = interquartile_mean(footprint)
        else : 
            footprint_kurto = None
            footprint_skew = None
            footprint_IQR = None
            footprint_IQM = None

        features["rec_footprint_kurto"] = footprint_kurto
        features["rec_footprint_skew"] = footprint_skew
        features["rec_footprint_IQR"] = footprint_IQR
        features["rec_footprint_IQM"] = footprint_IQM

         
    if "orientation_entropy" in features_list : 
        Ho,Hw,Phi = entropy_orientation(graph)
        features["Ho"] = Ho
        features["Hw"] = Hw
        features["Phi"] = Phi

    if "street_linearity" in features_list : 
        edges_lin_kurto,edges_lin_skew = edges_linearity(graph)
        features["edges_lin_kurto"] = edges_lin_kurto
        features["edges_lin_skew"] = edges_lin_skew

    if "meshedness" in features_list : 
        meshedness = compute_meshedness(graph)
        features["meshedness"] = meshedness

    if "intersection_ratio" in features_list: 
        ratio = intersection_ratio(graph)
        features["intersection_ratio"] = ratio

    if "deadends_ratio" in features_list:
        deadends_ratio = degree_ratio(graph,degree=1)
        features["deadends_ratio"] = deadends_ratio

    #Add Number of nodes/edges but not used in features at the end
    features["Nodes_number"] = len(graph.nodes())
    features["Edges_number"] = len(graph.edges())
    return features

def inspect_features(graph:nx.Graph,features_list:list[str])->nx.Graph:
    """Return the distribution of each features for each graph

    Args:
        G (nx.Graph): Input G
        features_list (list[str]): list of metrics to compute

    Returns:
        nx.Graph: embedded graph
    """

    features = {}

    if "betweenness" in features_list:
        offset = 1e-5
        BC_distr = betweenness_centrality(graph)
        features["BC_distr"] = BC_distr

    if "straightness" in features_list:
        SC_distr = straightness_centrality(graph)
        features["SC_distr"] = SC_distr

    if "circular_footprint" in features_list:

        footprint = circular_footprint(graph)

        if isinstance(footprint,np.ndarray): 
            hist, bin_edges = np.histogram(footprint,range=(0,1),bins=100,weights=np.ones(len(footprint)) / len(footprint))
            features["circular_footprint"] = hist
        else : 
            features["circular_footprint"] = 0

    if "orientation_entropy" in features_list : 
        Ho,Hw,Phi = entropy_orientation(graph)
        features["Ho"] = Ho
        features["Hw"] = Hw
        features["Phi"] = Phi
    if "rec_footprint" in features_list:

        footprint = rec_footprint(graph)

        if isinstance(footprint,np.ndarray): 
            features["rec_footprint"] = footprint
        else : 
            features["rec_footprint"] = 0
         
    if "orientation_entropy" in features_list : 
        Ho,Hw,Phi = entropy_orientation(graph)
        features["Ho"] = Ho
        features["Hw"] = Hw
        features["Phi"] = Phi


    if "meshedness" in features_list : 
        meshedness = compute_meshedness(graph)
        features["meshedness"] = meshedness

    if "intersection_ratio" in features_list: 
        ratio = intersection_ratio(graph)
        features["intersection_ratio"] = ratio

    return features

def clean_graph(graph:nx.Graph,features_list:list[str])-> nx.Graph:
    """Remove the unecessary node/edges attributes 

    Args:
        graph (nx.Graph): input raw graph
        features_list (list[str]): list of features to keep

    Returns:
        nx.Graph: cleaned graph
    """    
        # Remove unused features
    for node in graph.nodes():
        unused_node_keys = [key for key in graph.nodes()[node] if key not in features_list]
        for key in unused_node_keys:
            remove_attribute(graph, node, key)
        unused_edge_keys = [key for key in graph.in_edges(node).key()]
        print('unused edge keys',unused_edge_keys)


def remove_attribute(graph:nx.Graph,tnode:int,attr:str) -> nx.Graph:
    graph.nodes()[tnode].pop(attr,None)
    return graph

def make_node_df(graph:list[nx.Graph]) -> list[pd.DataFrame]:
    """Create dataframe from graph with node attributes

    Args:
        G (nx.Graph()): Input Graph
    
    Returns:
        list[pd.Dataframe] : list of graphs dataframes
    """    
    nodes = {}
    for node, attribute in graph.nodes(data=True):
        if not nodes.get('node'):
            nodes['node'] = [node]
        else:
            nodes['node'].append(node)

        for key, value in attribute.items():
            if not nodes.get(key):
                nodes[key] = [value]
            else:
                nodes[key].append(value)
    graph_df = pd.DataFrame.from_dict(nodes,orient='index').transpose()
    return graph_df
                

def interquartile_mean(arr):
    # Step 1: Sort the array
    sorted_arr = np.sort(arr)
    
    # Step 2: Find Q1 and Q3
    Q1 = np.percentile(sorted_arr, 25)
    Q3 = np.percentile(sorted_arr, 75)
    
    # Step 3: Identify data points between Q1 and Q3
    interquartile_data = sorted_arr[(sorted_arr >= Q1) & (sorted_arr <= Q3)]
    
    # Step 4: Compute the mean of these data points
    interquartile_mean = np.mean(interquartile_data)
    
    return interquartile_mean

def normalize_feat_df(graph_df:pd.DataFrame,feat_list:list[str]) -> pd.DataFrame:
    """Normalise the indicated features from a dataframe

    Args:
        graph_df (pd.DataFrame): Dataframe of the graph
        feat_list (list[str]): List of the features to normalize

    Returns:
        pd.DataFrame: Graph Dataframe with the normalized features
    """ 
    graph_df.iloc[feat_list] = graph_df.iloc[feat_list].apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    return graph_df

def normalize_feat_graph(graph:nx.Graph,feat_list:list[str]) -> nx.Graph:
    """Normalise the indicated features from a nx.Graph

    Args:
        graph (nx.Graph)): Graph
        feat_list (list[str]): List of the features to normalize

    Returns:
        nx.Graph: Graph with the normalized features
    """ 
    for feat in feat_list : 
        raw_feat = np.reshape(np.array([graph.nodes[node][feat] for node in graph.nodes()]),(-1,1))
        # Min-Max Normalization
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(raw_feat)
        for i, node in enumerate(graph.nodes()):
            graph.nodes[node][feat] = float(normalized_features[i][0])

    return graph



def circular_footprint(G:nx.Graph) -> Tuple:
    """Return the average and std of the distribution of the ratio between the area of each polygon (A) and 
    the area of the circumscribed circle (Ac)

    Args:
        G (nx.Graph): Input graph

    Returns:
        Tuple(int,int): average and std of the ratio distribution     
    """    
    gdf_node,gdf_edge = ox.utils_graph.graph_to_gdfs(
    G, nodes=True, edges=True, node_geometry=True,
    fill_edge_geometry=False)

    if not G.graph["crs"] == 3827: 
        G = ox.projection.project_graph(G, to_crs=3857, to_latlong=False)
    elif isinstance(G,nx.classes.multidigraph.MultiDiGraph) : 
        G = ox.convert.to_undirected(G)

    polygons = list(polygonize(gdf_edge.geometry.values))
    footprint = np.array([])
    gdf_polygons = gpd.GeoSeries(polygons)
    polygons_area = [polygon.area for polygon in polygons]

    if len(polygons) != 0 :
        footprint = mp.CircularCompactness(gdf_polygons, polygons_area).series.values
        return footprint
    else : 
        return None
    
def rec_footprint(G:nx.Graph) -> Tuple:
    """Return the average and std of the distribution of the ratio between the area of each polygon (A) and 
    the area of the circumscribed circle (Ac)

    Args:
        G (nx.Graph): Input graph

    Returns:
        Tuple(int,int): average and std of the ratio distribution     
    """    

    if not G.graph["crs"] == 3827: 
        G = ox.projection.project_graph(G, to_crs=3857, to_latlong=False)
    elif isinstance(G,nx.classes.multidigraph.MultiDiGraph) : 
        G = ox.convert.to_undirected(G)

    gdf_node,gdf_edge = ox.utils_graph.graph_to_gdfs(
    G, nodes=True, edges=True, node_geometry=True,
    fill_edge_geometry=True)

    #Combine all edges into a single MultiLineString
    polygons = list(polygonize(gdf_edge.geometry.values))

    gdf_polygons = gpd.GeoSeries(polygons)
    polygons_area = [polygon.area for polygon in polygons]
    if len(polygons) != 0 : 
        footprint = mp.Squareness(gdf_polygons).series.values
        return footprint
    else : 
        return None


def entropy_orientation(G:nx.Graph) -> float : 
    """Calculate undirected graph's orientation entropy

    Args:
        G (nx.Graph): input graph
    Returns:
        float: Orientation entropy
    """    
    if not G.graph["crs"] == 4326: 
        G = ox.projection.project_graph(G, to_crs=4326, to_latlong=True)
    if isinstance(G,nx.classes.multidigraph.MultiDiGraph) : 
        G = ox.convert.to_undirected(G)
    Gu = ox.add_edge_bearings(G) #Add compass bearing attributes to all graph edges.

    Ho = ox.bearing.orientation_entropy(Gu, num_bins=36, min_length=2, weight=None)
    Hw = ox.bearing.orientation_entropy(Gu, num_bins=36, min_length=2, weight="length")
    Hgrid = 1.386
    Hmax = 3.584
    Phi = 1 - ((Ho-Hgrid)/(Hmax-Hgrid))**2


    return Ho,Hw,Phi

def entropy_orientation_Phi(G:nx.Graph) -> float : 
    """Calculate undirected graph's orientation entropy

    Args:
        G (nx.Graph): input graph
    Returns:
        float: Orientation entropy
    """    
    if not G.graph["crs"] == 4326: 
        G = ox.projection.project_graph(G, to_crs=4326, to_latlong=True)
    if isinstance(G,nx.classes.multidigraph.MultiDiGraph) : 
        G = ox.convert.to_undirected(G)
    Gu = ox.add_edge_bearings(G) #Add compass bearing attributes to all graph edges.

    Ho = ox.bearing.orientation_entropy(Gu, num_bins=36, min_length=30, weight=None)
    Hw = ox.bearing.orientation_entropy(Gu, num_bins=36, min_length=30, weight="length")
    Hgrid = 1.386
    Hmax = 3.584
    Phi = 1 - ((Ho-Hgrid)/(Hmax-Hgrid))**2


    return Phi


def betweenness_centrality(G:nx.Graph) -> Tuple : 
    """Calculate skewness and kurtosis of BC distribution

    Args:
        G (nx.Graph): input graph
    Returns:
        tuple(float,float): skewness and kurtosis
    """    

    bc = nx.betweenness_centrality(ox.convert.to_digraph(G))
    nx.set_node_attributes(G, bc, "betweenness")
    nodes_BC = nx.get_node_attributes(G, "betweenness")
    BC_distr = []
    for _ , val in nodes_BC.items() :
        BC_distr.append(val)

    return np.array(BC_distr)

def straightness_centrality(G:nx.Graph) -> Tuple : 
    """Calculate skewness and kurtosis of straightness_centrality distribution

    Args:
        G (nx.Graph): input graph
    Returns:
        tuple(float,float): skewness and kurtosis
    """    

    if not G.graph["crs"] == 3827: 
        G = ox.projection.project_graph(G, to_crs=3857, to_latlong=False)

    G = mp.straightness_centrality(ox.convert.to_digraph(G),weight="length",name="straightness")
    nodes_SC = nx.get_node_attributes(G, "straightness")
    SC_distr = []
    for _ , val in nodes_SC.items() :
        SC_distr.append(val)
    return SC_distr

def edges_linearity(G:nx.Graph) -> list[float] : 
    """Compute the interquartile, mean and std of the edges linearity distribution


    Args:
        G (nx.Graph): input graph

    Returns:
        list[float]: [interquartile,mean,std]
    """    

    edges = ox.graph_to_gdfs(G,nodes=False,edges=True,node_geometry=False,fill_edge_geometry=True)
    edg_lin = mp.Linearity(edges).series.values
    kurto = kurtosis(edg_lin,fisher=False)
    skewness = skew(edg_lin)

    return kurto, skewness

def compute_meshedness(G:nx.Graph) -> float : 
    """Compute the meshedness coefficient of the planar graph


    Args:
        G (nx.Graph): input graph

    Returns:
        float : meshedness coeff
    """    

    Ne = len(G.edges)
    N = len(G.nodes)
    meshedness = (Ne - N + 1) / (2*N-5) 
    return meshedness

def intersection_ratio(G:nx.Graph) -> float : 
    """Compute the intersection_ratio of the planar graph


    Args:
        G (nx.Graph): input graph

    Returns:
        float : intersection ratio
    """    
    N1 = [val for (node,val) in G.degree].count(1)
    N3 = [val for (node,val) in G.degree].count(3)
    N2 = [val for (node,val) in G.degree].count(2)

    ratio = (N1 + N3)/(len(G.nodes)-N2)
    return ratio


def generate_df_emb(graphs_dir:Path) -> pd.DataFrame:
    """From pickle files of embedding graphs generate the dataframe with the column
    "label" corresponding to the name of the emb graph with its features vector in "features"
    column

    Args:
        graphs_dir (Path): graph path of the directory of embedded graphs

    Returns:
        pd.DataFrame: dataframe with vectors features and labels
    """
    embeddings_list = []
    for i, path in enumerate(graphs_dir.rglob("*.pck")) :
        label = path.stem        
        with open(path, 'rb') as file: 
            embedding = pickle.load(file)
        dict1 = {'label':label,'features':embedding}
        embeddings_list.append(dict1)
    embedding_df = pd.DataFrame(embeddings_list)

    return embedding_df

# Normalize the DataFrame
def normalize_dataframe(df:pd.DataFrame)->pd.DataFrame:
    """Normalize each columns and replace NaN values with 0

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: normalized and cleaned dataframe
    """    
    normalized_df = df.apply(zscore)
    return normalized_df



def plot_features(df: pd.DataFrame, normalise: bool, embedding: bool) -> None:
    """
    Plot each feature of each input graph

    Args:
        df (pd.DataFrame): features graphs dataframe
        normalise (bool): If True, normalise features distribution before plot
        embedding (bool): If True, plot all the embedding features, if not plot the actual raw features (distributions,..)
    """

    Nb_graphs = len(df)
    features_labels = df.iloc[0]["features"].keys()
    Nb_features = len(features_labels)
    features_df = pd.DataFrame({feat: df["features"].apply(lambda x: x[feat]).tolist() for feat in features_labels})
    features_df = features_df.fillna(0)  # Clean NaN values
    
    colors = plt.cm.get_cmap('tab10', Nb_graphs)

    if embedding:
        if normalise:
            features_df = normalize_dataframe(features_df)

        plt.figure(figsize=(12, 8))
        for graph_idx in range(Nb_graphs):
            for feature_idx, feature_name in enumerate(features_df.columns):
                if pd.notnull(features_df.loc[graph_idx, feature_name]):
                    x = features_df.loc[graph_idx, feature_name]
                    y = feature_name
                    color = colors(graph_idx)
                    graph_label = df.iloc[graph_idx]["label"]
                    plt.scatter(x, y, color=color, alpha=0.7, label=graph_label if feature_idx == 0 else "")
        
        plt.xlabel('Value of Features for Each Graph')
        plt.ylabel('Name of Features')
        plt.title('Scatter Plot of Feature Values for Each Graph')
        plt.grid(True)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title='Graphs', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

    else:
        if normalise:
            scaler = MinMaxScaler()
            features_df = pd.DataFrame({feat: df["features"].apply(lambda x: np.squeeze(scaler.fit_transform(np.reshape(x[feat], (-1, 1))))) for feat in features_labels})
        
        fig, axs = plt.subplots(Nb_features, 1, figsize=(12, 8))
        
        if Nb_features == 1:
            axs = [axs]
        
        for i, feature_name in enumerate(features_df.columns):
            for graph_idx in range(Nb_graphs):
                feat = features_df.loc[graph_idx, feature_name]
                color = colors(graph_idx)
                graph_label = df.iloc[graph_idx]["label"]
                if not isinstance(feat, float):
                    axs[i].hist(feat, color=color, alpha=0.7, bins=50, label=graph_label, weights=np.ones(len(feat)) / len(feat))
                    axs[i].set_title(feature_name)
                    axs[i].set_xlabel(f'Values of {feature_name}')
                    axs[i].set_ylabel('Frequency')
                    axs[i].grid(True)
        
        for ax in axs:
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.5)
        plt.show()



def plot_clusters(training_df: pd.DataFrame, predict_df: pd.DataFrame,normalise: bool,features_selection:bool,features_list: list,compare_cluster_dist: bool,
                  Kbest_feats:int, n_neighbors:int,min_dist:float,n_components:int,random_state:None,metric:str) -> None:
    """
    Plot each feature of each input graph

    Args:
        training_df (pd.DataFrame): features real graphs dataframe
        predict_df (pd.DataFrame): features toy graphs dataframe
        normalise (bool): If True, normalise features distribution before plot
        features_selection(bool): If True, selection features according to SelecKBest
        features_list (list): List of strings containing used features for embedding
        compare_cluster_dist (bool): If True, plot the distances between clusters in high and low dimensional space embedding
        Kbest_feats (int): Number of best features to select
        n_neighbors (int): Size of local neighborhood UMAP will look
        min_dist (float): Minimum distance apart that points are allowed to be in the low dimensional representation
        n_components (int): DFimensionality of the reduced dimension space 
        random_state: int, RandomState instance or None, optional (default: None)
                    If int, random_state is the seed used by the random number generator;
                    If RandomState instance, random_state is the random number generator;
                    If None, the random number generator is the RandomState instance used
                    by `np.random`.
        metric (str): metric used in UMAP projection
    """


    features_df = pd.DataFrame({feat: training_df["features"].apply(lambda x: x[feat]).tolist() for feat in features_list})
    training_features_df = features_df.fillna(0)  # Clean NaN values

    features_df = pd.DataFrame({feat: predict_df["features"].apply(lambda x: x[feat]).tolist() for feat in features_list})
    toy_features_df = features_df.fillna(0)  # Clean NaN values

    all_df = pd.concat([training_df,predict_df],ignore_index=True)
    all_df_features = pd.concat([training_features_df,toy_features_df])
    all_df = pd.concat([training_df,predict_df])


    if normalise:
        # Fit the StandardScaler on the training data
        scaler = StandardScaler()
        features_to_norm = training_features_df.columns
        #features_to_norm = features_to_norm.drop(['Ho', 'Hw', 'Phi'])
        training_features_df[features_to_norm] = scaler.fit_transform(training_features_df[features_to_norm]) 
        toy_features_df[features_to_norm] = scaler.transform(toy_features_df[features_to_norm])

    #TRAIN UMAP
    trans = UMAP(
        n_neighbors=n_neighbors, 
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric=metric,
    ).fit(training_features_df.values)

    #EMBEDDING
    real_embeddings = trans.transform(training_features_df.values)
    toy_embeddings = trans.transform(toy_features_df.values)


    high_dim_embeddings = np.concatenate((real_embeddings,toy_embeddings))

    #CLUSTERING
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    high_dim_clusters = clusterer.fit(high_dim_embeddings)
    Nb_clusters = len(np.unique(high_dim_clusters.labels_))

    #COMPUTE 2D EMBEDDING AND CLUSTERING    
    if n_components != 2 :
        trans = UMAP(
            n_neighbors=n_neighbors, 
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
            metric=metric,
        ).fit(training_features_df.values)

        real_embeddings = trans.transform(training_features_df.values)
        toy_embeddings = trans.transform(toy_features_df.values)


        
        low_dim_embeddings = np.concatenate((real_embeddings,toy_embeddings))

        clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
        low_dim_clusters = clusterer.fit(low_dim_embeddings)
        Nb_clusters = len(np.unique(low_dim_clusters.labels_))

    ### PLOT 

    #Compute correlation of intra/inter clusters distances
    if compare_cluster_dist and n_components != 2 : 
        compare_cluster_distances(high_dim_embeddings, high_dim_clusters.labels_, low_dim_embeddings, low_dim_clusters.labels_,metric)

    if features_selection : 
        # Create and fit selector
        selector = SelectKBest(f_classif, k=Kbest_feats)
        selector.fit(all_df_features, high_dim_clusters.labels_)
        # Get columns to keep and create new dataframe with those only
        cols_idxs = selector.get_support(indices=True)
        features_df = features_df.iloc[:,cols_idxs]
        low_dim_embeddings = features_df.values
    
    color_palette = sns.color_palette("husl", Nb_clusters).as_hex()
    markers = ["o", "x"]


    if not features_selection or Kbest_feats != 3:
        plt.figure(figsize=(25, 10))
        detailed_legend = []
        ax = plt.gca()  # Get the current axis
        if features_selection:
            clusters = high_dim_clusters
            embeddings = low_dim_embeddings
        else:
            if n_components == 2:
                clusters = high_dim_clusters
                embeddings = high_dim_embeddings
            else:
                clusters = low_dim_clusters
                embeddings = low_dim_embeddings

        for i, (point, label) in enumerate(zip(embeddings, all_df['label'])):
            x, y = point
            color = color_palette[clusters.labels_[i]]
            marker = 's' if 'bc' in label or 'toy' in label else 'o' 
            plt.scatter(x, y, color=color, marker=marker, zorder=3 if 'toy' in label else 2)
            detailed_legend.append(plt.Line2D([0], [0], marker=marker, color='w', label=f'{i + 1}: {label}', markerfacecolor=color, markersize=10))

        # Création des légendes pour les marqueurs
        legend_markers = [plt.Line2D([], [], color='black', marker='s', markersize=10, label='Toy'),
                        plt.Line2D([], [], color='black', marker='o', markersize=10, label='Real')]

        # Ajouter la légende principale au graphique
        legend1 = ax.legend(handles=detailed_legend, bbox_to_anchor=(1.15, 0), loc='lower left', borderaxespad=0.,
                            title="Graph Labels", frameon=True, framealpha=1, edgecolor='black')
        ax.add_artist(legend1)  # Ajouter la première légende manuellement

        # Ajouter la légende des marqueurs au graphique
        legend2 = ax.legend(handles=legend_markers, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                            title="Marker Symbols", frameon=True, framealpha=1, edgecolor='black')

        if features_selection:
            plt.xlabel(f'Dimension {features_df.columns[0]}')
            plt.ylabel(f'Dimension {features_df.columns[1]}')
        else:
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')

        plt.title('Clustered Projected Embeddings with UMAP+HDBSCAN')
        plt.grid()
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.show()
    else : 
        fig = plt.figure(figsize=(25, 10))
        ax = fig.add_subplot(111, projection='3d')
        detailed_legend = []

        # Define marker styles and descriptions
        marker_styles = {'Real': 'o', 'Toy': 's'}

        # Initialize a flag to check if any points are plotted
        has_real = False
        has_toy = False

        for i, (point, label) in enumerate(zip(low_dim_embeddings, all_df['label'])):
            x, y, z = point  # Assuming embeddings have 3 dimensions
            clusters = low_dim_clusters if n_components != 2 else high_dim_clusters
            color = color_palette[clusters.labels_[i]]
            
            # Determine marker based on label
            if 'bc' in label or 'toy' in label:
                marker = marker_styles['Real']  # Circle for Real
                has_real = True
            else:
                marker = marker_styles['Toy']  # Square for Toy
                has_toy = True

            ax.scatter(x, y, z, color=color, marker=marker)
            ax.text(x, y, z, label if 'toy' in label else str(i + 1), color='black', fontsize=12)

            # Adding the detailed legend for individual points
            detailed_legend.append(plt.Line2D([0], [0], marker=marker, color='w', label=f'{i + 1}: {label}', markerfacecolor=color, markersize=10))

        # Create legend for markers if both types are present
        marker_legend = []
        if has_real:
            marker_legend.append(plt.Line2D([], [], color='black', marker='o', markersize=10, label='Real'))
        if has_toy:
            marker_legend.append(plt.Line2D([], [], color='black', marker='s', markersize=10, label='Toy'))

        # Add main legend to the plot
        legend1 = ax.legend(handles=detailed_legend, bbox_to_anchor=(1.15, 0), loc='lower left', borderaxespad=0.,
                            title="Graph Labels", frameon=True, framealpha=1, edgecolor='black')
        ax.add_artist(legend1)  # Add the first legend manually

        # Add marker legend to the plot if markers are present
        if marker_legend:
            legend2 = ax.legend(handles=marker_legend, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                                title="Marker Symbols", frameon=True, framealpha=1, edgecolor='black')

        # Set labels and title
        ax.set_xlabel(f'Dimension {features_df.columns[0]}')
        ax.set_ylabel(f'Dimension {features_df.columns[1]}')
        ax.set_zlabel(f'Dimension {features_df.columns[2]}')
        ax.set_title('Clustered Projected Embeddings with UMAP+HDBSCAN')

        ax.grid()
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.show()




def compare_cluster_distances(high_dim_data: np.ndarray, labels_high_dim: np.ndarray, umap_2d: np.ndarray, labels_2d: np.ndarray,metric:str) -> None:
    """
    Compare and visualize cluster distances between high-dimensional data and its 2D UMAP projection.

    Parameters:
    - high_dim_data: np.ndarray
        The original high-dimensional dataset.
    - labels_high_dim: np.ndarray
        Cluster labels for the high-dimensional dataset obtained using HDBSCAN.
    - umap_2d: np.ndarray
        The 2D UMAP projection of the high-dimensional dataset.
    - labels_2d: np.ndarray
        Cluster labels for the 2D UMAP projection obtained using HDBSCAN.
    - metric: str
        Used metric in UMAP projection
    Returns:
    None. The function prints distance comparisons and displays scatter plots for cluster visualization.
    """

    # Calculate pairwise distance matrices
    if metric == 'manhattan' : 
        dist_high_dim = manhattan_distances(high_dim_data)
        dist_2d = manhattan_distances(umap_2d)
    else : 
        dist_high_dim = pairwise_distances(high_dim_data)
        dist_2d = pairwise_distances(umap_2d)
    # Calculate within-cluster and between-cluster distances for both datasets
    within_high_dim, between_high_dim = compute_distances(dist_high_dim, labels_high_dim)
    within_2d, between_2d = compute_distances(dist_2d, labels_2d)

    # Correlation between high-dimensional and 2D distances
    correlation = np.corrcoef(dist_high_dim.flatten(), dist_2d.flatten())[0, 1]

    # Plotting the comparison
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot high-dimensional clusters in 2D space
    scatter = ax[0].scatter(umap_2d[:, 0], umap_2d[:, 1], c=labels_high_dim, cmap='viridis', s=50, alpha=0.7)
    ax[0].set_title('Clusters in High-Dimensional Space (2D Projection)')
    ax[0].set_xlabel('UMAP Dimension 1')
    ax[0].set_ylabel('UMAP Dimension 2')
    legend1 = ax[0].legend(*scatter.legend_elements(), title="Clusters")
    ax[0].add_artist(legend1)

    # Plot 2D UMAP clusters
    scatter = ax[1].scatter(umap_2d[:, 0], umap_2d[:, 1], c=labels_2d, cmap='viridis', s=50, alpha=0.7)
    ax[1].set_title('Clusters in 2D UMAP Projection')
    ax[1].set_xlabel('UMAP Dimension 1')
    ax[1].set_ylabel('UMAP Dimension 2')
    legend2 = ax[1].legend(*scatter.legend_elements(), title="Clusters")
    ax[1].add_artist(legend2)

    # Plotting the distance comparisons and correlation
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    indices = np.arange(2)

    within_distances = [within_high_dim, within_2d]
    between_distances = [between_high_dim, between_2d]

    ax2.bar(indices, within_distances, bar_width, label='Within-Cluster')
    ax2.bar(indices + bar_width, between_distances, bar_width, label='Between-Cluster')

    ax2.set_title('Distance Comparisons')
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Mean Distance')
    ax2.set_xticks(indices + bar_width / 2)
    ax2.set_xticklabels(['High-Dim', '2D UMAP'])
    ax2.legend()

    # Add correlation information
    correlation_text = f'Correlation: {correlation:.2f}'
    ax2.text(1, max(max(within_distances), max(between_distances)) * 0.95, correlation_text, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    # Print the distance comparisons
    print(f'High-Dimensional Space: Within-Cluster = {within_high_dim}, Between-Cluster = {between_high_dim}')
    print(f'2D Projection: Within-Cluster = {within_2d}, Between-Cluster = {between_2d}')
    print(f'Correlation between high-dimensional and 2D distances: {correlation}')



def compute_distances(dist_matrix: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Compute within-cluster and between-cluster distances.

    Parameters:
    - dist_matrix: np.ndarray
        Pairwise distance matrix of the dataset.
    - labels: np.ndarray
        Cluster labels for the dataset.

    Returns:
    - within_cluster_mean: float
        Mean within-cluster distance.
    - between_cluster_mean: float
        Mean between-cluster distance.
    """
    unique_labels = np.unique(labels[labels >= 0])  # Ignore noise points with label -1
    within_cluster_dists = []
    between_cluster_dists = []

    for label in unique_labels:
        cluster_mask = labels == label
        within_cluster_dists.append(np.mean(dist_matrix[cluster_mask][:, cluster_mask]))
        for other_label in unique_labels:
            if label != other_label:
                other_cluster_mask = labels == other_label
                between_cluster_dists.append(np.mean(dist_matrix[cluster_mask][:, other_cluster_mask]))

    return np.mean(within_cluster_dists), np.mean(between_cluster_dists)



def threshold_filter(G, ratio,attribute) -> nx.Graph:
    """Return the simplified version of the graph based on the ratio of the tops nodes metric attribute

    Args:
        G (_type_): _description_
        ratio (_type_): _description_
        attribute (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """    
    # Ensure the ratio is between 0 and 1
    if not 0 <= ratio <= 1:
        raise ValueError("Ratio must be between 0 and 1")
    
    # Sort nodes by the specified attribute in ascending order
    sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1].get(attribute, float('inf')))
    
    # Determine the number of nodes to remove
    num_nodes_to_remove = int(len(sorted_nodes) * ratio)
    
    # Extract the nodes to remove
    nodes_to_remove = [node for node, _ in sorted_nodes[:num_nodes_to_remove]]
    
    # Remove the nodes from the graph
    G.remove_nodes_from(nodes_to_remove)
    
    return G

def convert_to_multidigraph_with_keys(G,crs) -> nx.MultiDiGraph:
    # Initialize a MultiDiGraph
    MG = nx.MultiDiGraph()
    
    # Copy nodes and node attributes
    for node, data in G.nodes(data=True):
        MG.add_node(node, **data)
    
    # Copy edges and edge attributes, ensuring unique keys
    for u, v, data in G.edges(data=True):
        MG.add_edge(u, v, key=0, **data)  # Key is set to 0 by default, modify if necessary
    MG.graph["crs"] = crs
    return MG


def plot_length(graphs_dict:dict) -> None: 
    weights_list = []
    scaler = MinMaxScaler()
    for label, graph in graphs_dict.items():
        data = np.array(weights(graph)).reshape(-1,1)
        data = scaler.fit_transform(data)
        plt.hist(weights(graph),bins=50)
        plt.title(label)
        plt.show()

def toyness_scores(graphs_dict: dict,metric_names:list[str]) -> None : 

    # Dynamically get the metric functions from the names
    metrics = [globals()[name] for name in metric_names]

    # Prepare data for the dataframe
    data = []
    #plot_length(graphs_dict)

    for label, graph in graphs_dict.items():
        metric_values = []
        for metric in metrics:
            if metric == average_clustering_coefficient :
                metric_values.append(-metric(graph))
            else:
                metric_values.append(metric(graph))
        data.append([label] + metric_values)

    columns = ['Graph Label'] + [name for name in metric_names]
    df = pd.DataFrame(data, columns=columns)

    # Compute ranks for each metric (feature) within each graph
    ranked_df = df.copy()
    print('graph label index : ',ranked_df['Graph Label'])
    for column in metric_names:
        ranked_df[column + ' Rank'] = ranked_df[column].rank(method='average', ascending=False)
        print('column : ', column)
        print('unweighted ranked column',ranked_df[column + ' Rank'])

    # Define the weighting system
    weights = [0.95 ** i for i in range(len(metric_names))]

    # Normalize ranks and apply weights
    for i, column in enumerate(metric_names):
        ranked_df[column + ' Normalized Rank'] = rankdata(ranked_df[column + ' Rank'])/len(graphs_dict)
        ranked_df[column + ' Weighted Normalized Rank'] = ranked_df[column + ' Normalized Rank'] * weights[i]
        print('column : ', column)
        print('weighted ranked column',ranked_df[column + ' Rank'])

    # Compute average weighted normalized ranks for each graph
    ranked_df['Average Weighted Normalized Rank'] = ranked_df[[column + ' Weighted Normalized Rank' for column in metric_names]].mean(axis=1)

    # Plotting average weighted normalized rank
    print('Average Weighted Normalized Rank', ranked_df['Average Weighted Normalized Rank'])
    color = ['red' if '_bc' in label or 'toy' in label else 'black' for label in ranked_df['Graph Label']]
    plt.scatter(ranked_df['Graph Label'], ranked_df['Average Weighted Normalized Rank'], marker='s', c=color)
    # Adding a custom legend
    red_patch = plt.Line2D([0], [0], marker='s', color='w', label='Toy Graph', markerfacecolor='red', markersize=10)
    black_patch = plt.Line2D([0], [0], marker='s', color='w', label='Real Graph', markerfacecolor='black', markersize=10)

    plt.legend(handles=[red_patch, black_patch])
    # Adding labels and title
    plt.xlabel('Graph Label')
    plt.ylabel('Weighted Normalized Rank')
    plt.title('Weighted Normalized Ranks of Graphs')
    plt.xticks(rotation=45)
    plt.grid(True)
    # Show plot
    plt.tight_layout()
    plt.show()


def toyness_scores_metrics(graphs_dict: dict, metric_names: list[str], threshold: float) -> None:
    # Dynamically get the metric functions from the names
    metrics = [globals()[name] for name in metric_names]

    # Prepare data for the dataframe
    data = []

    for label, graph in graphs_dict.items():
        metric_values = []
        for metric in metrics:
            if metric == average_clustering_coefficient:
                metric_values.append(-metric(graph))
            else:
                metric_values.append(metric(graph))
        data.append([label] + metric_values)

    columns = ['Graph Label'] + [name for name in metric_names]
    df = pd.DataFrame(data, columns=columns)

    # Compute ranks for each metric (feature) within each graph
    ranked_df = df.copy()
    for column in metric_names:
        ranked_df[column + ' Rank'] = ranked_df[column].rank(method='average', ascending=False)

    # Define the weighting system
    weights = [0.95 ** i for i in range(len(metric_names))]

    # Normalize ranks and apply weights
    for i, column in enumerate(metric_names):
        ranked_df[column + ' Normalized Rank'] = rankdata(ranked_df[column + ' Rank']) / len(graphs_dict)
        ranked_df[column + ' Weighted Normalized Rank'] = ranked_df[column + ' Normalized Rank'] * weights[i]

    # Compute average weighted normalized ranks for each graph
    ranked_df['Average Weighted Normalized Rank'] = ranked_df[[column + ' Weighted Normalized Rank' for column in metric_names]].mean(axis=1)

    # Label the graphs: 0 if "toy" or "bc" in label, 1 otherwise
    ranked_df['Label'] = ranked_df['Graph Label'].apply(lambda x: 0 if 'toy' in x or 'bc' in x else 1)

    # Predict labels based on the threshold
    ranked_df['Predicted Label'] = ranked_df['Average Weighted Normalized Rank'].apply(lambda x: 1 if x >= threshold else 0)

    # Evaluate the threshold
    y_true = ranked_df['Label']
    y_pred = ranked_df['Predicted Label']

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, ranked_df['Average Weighted Normalized Rank'])
    roc_auc = auc(fpr, tpr)

    # Print evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {roc_auc:.4f}')

    # Plotting average weighted normalized rank
    color = ['red' if label == 0 else 'black' for label in ranked_df['Label']]
    plt.scatter(ranked_df['Graph Label'], ranked_df['Average Weighted Normalized Rank'], marker='s', c=color)
    # Adding a custom legend
    red_patch = plt.Line2D([0], [0], marker='s', color='w', label='Toy Graph', markerfacecolor='red', markersize=10)
    black_patch = plt.Line2D([0], [0], marker='s', color='w', label='Real Graph', markerfacecolor='black', markersize=10)

    plt.legend(handles=[red_patch, black_patch])
    # Adding labels and title
    plt.xlabel('Graph Label')
    plt.ylabel('Weighted Normalized Rank')
    plt.title('Weighted Normalized Ranks of Graphs')
    plt.xticks(rotation=45)
    plt.grid(True)
    # Show plot
    plt.tight_layout()
    plt.show()

    # Plotting ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Plotting Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Predicted Toy/Extracted', 'Predicted Real'], yticklabels=['Actual Toy/Extracted', 'Actual Real'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def average_degree(graph:nx.Graph) -> float:
    return np.mean(degrees(graph))

def vmr(G:nx.Graph) -> nx.Graph : 
    scaler = MinMaxScaler()
    data = np.array(weights(G)).reshape(-1,1)
    data = scaler.fit_transform(data)
    data = np.array([x[0] for x in data])
    vmr = np.var(data) / np.mean(data)
    return vmr
def IQR_weights(G:nx.Graph) -> nx.Graph :
    scaler = MinMaxScaler()
    data = np.array(weights(G)).reshape(-1,1)
    data = scaler.fit_transform(data)
    data = np.array([x[0] for x in data])
    IQR = scipy.stats.iqr(data)
    return IQR
def degrees(G:nx.Graph, weight=None) -> list :
    return list(dict(G.degree(weight=weight)).values())

def density(graph:nx.Graph) -> float :
    return round(nx.density(graph), 4)

def average_clustering_coefficient(G:nx.Graph) -> float:
    node_clustering = ig.Graph.from_networkx(G).transitivity_local_undirected(mode="nan")
    return np.mean([x for x in node_clustering if isinstance(x, float) and not np.isnan(x)])

def reachability(G:nx.Graph) -> float : 
    r = 0
    for c in [len(component) for component in nx.connected_components(G)]:
        r += c*(c-1)
    return r/(len(G)*(len(G) - 1))

def number_connected_components(G) -> int :
    return nx.number_connected_components(G)


def weight_entropy(graph:nx.Graph) -> float :
    scaler = MinMaxScaler()
    data = weights(graph)
    data = np.array(weights(graph)).reshape(-1,1)
    data = scaler.fit_transform(data)
    data = np.array([x[0] for x in data])
    return entropy(data)


def weights(G:nx.Graph) -> list :
    data = list(nx.get_edge_attributes(G, 'length').values())
    edges = [edge for edge in data if edge >30]
    return edges

def kurtosis_weights(G:nx.Graph) -> float :
    scaler = MinMaxScaler()
    data = np.array(weights(G)).reshape(-1,1)
    data = scaler.fit_transform(data)
    data = np.array([x[0] for x in data])
    return kurtosis(data)

def skewness_weights(G:nx.Graph) -> float :
    scaler = MinMaxScaler()
    data = np.array(weights(G)).reshape(-1,1)
    data = scaler.fit_transform(data)
    data = np.array([x[0] for x in data])
    return skew(data)

def gini_coeff(graph:nx.Graph) -> float :
    scaler = MinMaxScaler()
    data = np.array(weights(graph)).reshape(-1,1)
    data = scaler.fit_transform(data)
    values = np.array([x[0] for x in data])
    sorted_values = np.sort(values)
    n = len(values)
    cumulative_values = np.cumsum(sorted_values, dtype=float)
    cumulative_share = cumulative_values / cumulative_values[-1]
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * cumulative_share) / n - (n + 1)) / n
    return gini

def diptest_fn(graph:nx.Graph) -> float :
    scaler = MinMaxScaler()
    data = np.array(weights(graph)).reshape(-1,1)
    data = scaler.fit_transform(data)
    data = np.array([x[0] for x in data])
    # only the dip statistic
    dip = diptest.dipstat(data)
    
    # both the dip statistic and p-value
    dip, pval = diptest.diptest(data)
    return pval

def ks_test(graph:nx.Graph) -> float :
    scaler = MinMaxScaler()
    data = np.array(weights(graph)).reshape(-1,1)
    data = scaler.fit_transform(data)
    data = np.array([x[0] for x in data])
    _, p = kstest(data,'norm')
    return p>0.05

def std_test(graph:nx.Graph) -> float : 
    scaler = MinMaxScaler()
    data = np.array(weights(graph)).reshape(-1,1)
    data = scaler.fit_transform(data)
    data = np.array([x[0] for x in data])
    s = tstd(data)
    return s

def shapiro_test(graph:nx.Graph) -> float : 
    scaler = MinMaxScaler()
    data = np.array(weights(graph)).reshape(-1,1)
    data = scaler.fit_transform(data)
    data = np.array([x[0] for x in data])
    _,p = shapiro(data)
    return p 


def entropy_weight(graph:nx.Graph) -> float :
    scaler = MinMaxScaler()
    data = np.array(weights(graph)).reshape(-1,1)
    data = scaler.fit_transform(data)
    data = np.array([x[0] for x in data])
    return entropy(data)