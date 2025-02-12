# UrbanToyGraph

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![GitHub License](https://img.shields.io/github/license/csebastiao/UrbanToyGraph)](https://github.com/csebastiao/UrbanToyGraph/blob/main/LICENSE)



## General Pipeline

The general pipeline for clustering urban graphs is as follows: 
- Run 'graph_location_extractor.py' to generate the real graphs
- The graphs obtained are raw, to clean them you need to convert them into gpkg format by running `graphml_to_gpkg.py`, using a software such as QGIS, then convert the cleaned gpkg back into graphml by running `gpkg_to_graphml.py`.
- Run `graph_constructor.py` to generate the template toy graph
- Run `backbone_extractor.py` to generate the raw backbones extracted from the real cleaned graphs. You can also clean them up using the same procedure as above.
- Put the template toy graphs and the cleaned backbones in the same folder (for example 'toy_graphs')
- Run `graph_embedding.py` to generate the embeddings for the toy and real graphs.
- Run `graph_projection.py` to display the 2D projection with UMAP and the clusters identified with HDBSCAN.


To modify all the access paths and parameters, please refer to the indications below in the available "Functionalities".

## Installation

First clone the repository in the folder of your choosing:

```
git clone https://github.com/Arthurondeau/Toy_Graph_Motifs.git
```

Locate yourself within the cloned folder, and create a new virtual environment. You can either create a new virtual environment then install the necessary dependencies with `pip` using the `requirements.txt` file:

```
pip install -r requirements.txt
```

Or create a new environment with the dependencies with `conda` or `mamba` using the `setup.sh` file based on `requirements.txt` file :


```
source setup.sh
```

Or create a new environment with the dependencies with `conda` or `mamba` using the `environment.yml` file:

```
mamba env create -f environment.yml
```


Once your environment is ready, you can locally install the package using:

```
pip install -e .
```


## Functionalities

### Hydra 

All parameters and model's parameters are defined in config file with hydra library (allow to automatically instantiate models classes)
Hydra config doc can be found here for additional details https://hydra.cc/docs/intro/

### Extract real graphs

Using the 'scripts/graph_location_extractor' with the parameters located in 'conf/location_graph.yaml' you can generate an graph from OSM. The graphs are saved in `.graphml` format in the 
mentioned directory in the `.yaml` conf file.

### Create generated toy graphs 
There are two types of graphs : either the customizable spatial ones and the ones extracted from real graphs.

#### Create customizable spatial graph

Using the functions located in `utg/create_graph`, you can create spatial graphs, with non-intersecting edges having a geometry attribute. All graph can be created without additional arguments, but can be customized. Graph can be made osmnx-compatible using `utg.utils.make_osmnx_compatible`.

Here are some examples of graph made using `create_graph` functions. The plots are made and saved using `utg.utils.plot_graph`.The graphs are saved in `.graphml` format using `utg.utils.save_graph`. All are made in the script `graph_constructor.py`. Since we have a geometry attribute, graph saved need to be loaded using `utg.utils.load_graph`, to transform WKT-string to shapely geometry. All graph files and their picture are located in the mentioned directory in the `graph_constructor/defaults.yaml` conf file:

- Barcelona ![Barcelona](template_graph/barcelona.png)
- Bridge large ![Bridge large](template_graph/bridge_large.png)
- Concentric large ![Concentric large](template_graph/concentric_large.png)
- Fractaler cross ![Fractaler cross](template_graph/fractaler_cross.png)

#### Extract toy from real graphs

Run the script `backbone_extractor.py` to generate the extracted toy graphs. All the parameters and paths directories are indicated in the conf file `backbone/defaults.yaml`.

### Add or remove edges

To add some noise in those perfectly geometrical graph you can use `utg.create_graph.add_random_edges` and `utg.create_graph.remove_random_edges`. These functions should work for any spatial graph having `x` and `y` attributes on every nodes.

⚠️ Be careful, `utg.create_graph.add_random_edges` is still WIP. It should not add forbidden edges if every edges are straight. For a concentric graph, use `utg.create_graph.create_concentric_graph(straight_edges=True)` to ensure straight edges. Even with straight edges, some edges that should be possible to add might not be added, because this function is based on Voronoi cells to find visible neighbors to connect to, limiting the choices to the closest neighbors.


### Clean the graphs

#### Convert to gpkg from graphml
Using the  `graphml_to_gpkg.py` with the parameters located in `conf/clean_graph/defaults.yaml` you can generate an graph from OSM. The graphs are saved in `.graphml` format in the 
mentioned directory in the `.yaml` conf file.

#### Convert to graphml from gpkg
Using the  `gpkg_to_graphml.py` with the parameters located in `conf/clean_graph/defaults.yaml` you can generate an graph from OSM. The graphs are saved in `.graphml` format in the 
mentioned directory in the `.yaml` conf file.

### Compute embeddings 
Using the  `graph_embedding.py` with the parameters located in `conf/embedding/defaults.yaml` you can generate the embeddings. The embeddings are saved in `.pck` format in the 
mentioned directory in the `.yaml` conf file.

### UMAP Projection + HDSCAN clustering
From the embeddings, you can project them into a 2D space with UMAP with the identified clusters thanks to HDSCAN algorithm.
To do that, run `graph_projection.py`. 
Note : here the UMAP is trained with the real graphs, thus the toy graphs are projected on fixed space.

### Toyness score
Run `graph_toyness_score.py` to compute the toyness score for each graph based on the metrics indicated in the `conf/toyness_score/defaults.yaml` file.



