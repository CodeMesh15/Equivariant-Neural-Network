# -*- coding: utf-8 -*-
"""Understanding the Netowrkx.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MGuppC8Lyez4efmG6v9CrRklAMPJEfyB

# Introducing the networkx
"""

!pip install networkx
import networkx as nx

"""# Making the first Graph"""

G=nx.Graph()

nx.draw(G)

G.add_node(1)

G.add_nodes_from([2,3])
nx.draw(G)

G.add_edge(1,2)

nx.draw(G)

G.add_edges_from([(1,2),(2,3),(3,1)])
nx.draw(G)

G.add_nodes_from([4,5,6,7,8])
nx.draw(G)

G.add_edges_from([(5,1),(6,7)])
nx.draw(G)

G.add_edges_from([(4,3),(7,3)])
nx.draw(G)

print(G.number_of_nodes())
print(G.number_of_edges())

G.nodes

G.edges

G.graph

"""# JSON"""

from networkx.readwrite import json_graph
json_data=json_graph.node_link_data(G)
json_data

json_graph=json_graph.node_link_graph(json_data)
json_graph

nx.draw(json_graph)

G=json_graph
G.add_nodes_from([9,10])
nx.draw(G)

G.remove_node(10)
nx.draw(G)

G.order()

G.clear()

"""# Function"""

G.add_edges_from([(1,2),(2,3),(3,1)])
G.add_node('spam')
G.add_nodes_from('spam')
nx.draw(G)

from networkx.classes.function import density,degree,degree_histogram
print(density(G))
print(degree(G))
print(degree_histogram(G))

"""# Neighbors"""

from networkx.classes.function import neighbors
for node in G:
  print(node)
  print(list(G.neighbors(node)))

"""# Path Graphs"""

G=nx.path_graph(4)
nx.draw(G)
print(nx.is_weighted(G))
print(nx.is_directed(G))

"""# Directional Graphs"""

G=nx.DiGraph()
print(nx.is_weighted(G))
print(nx.is_directed(G))
G.add_edge(1,2,weight=2)
G.add_edge(2,1,weight=3)
G.add_edge(2,2,weight=3)
print(nx.is_weighted(G))
print(nx.is_directed(G))
nx.draw(G)

G.add_edges_from([(1,3),(3,2)])
nx.draw(G)

G.edges

G.nodes

for node in G.nodes:
  print(node)
  print(list(G.neighbors(node)))

G.number_of_nodes()

G.number_of_edges()

"""# Adjacency Matrix"""

A=nx.adjacency_matrix(G)
print(A.todense())

A.setdiag(A.diagonal()*2)
print(A.todense())

A=nx.to_numpy_array(G)
print(A)

A.diagonal()

"""#MultiDirected Graph"""

G=nx.MultiDiGraph()
G.add_edge(1,2,key='a',weight=2)
G.add_edge(1,2,key='b',weight=3)
nx.draw(G)
print(G.nodes)
print(G.edges)
d=nx.to_dict_of_dicts(G)
print(d[1][2]['a']['weight'])
print(d)

"""# Making the edge List"""

G=nx.Graph([('A','B',{'cost':5,'weight':6}),
 ('C','D',{'cost':4,'weight':7})
])
nx.draw(G)
print(G.edges)
print(G.nodes)
df=nx.to_pandas_edgelist(G)
print(df)
df=nx.to_pandas_edgelist(G,nodelist=['A','B'])
A=nx.adjacency_matrix(G)
print(A.todense())
df

"""# MultiGraph and Its edge List"""

G=nx.MultiGraph()
G.add_edge('A','B',cost=1)
G.add_edge('A','B',cost=9)
# G.add_edge('A','B',cost=27)
nx.draw(G)
print(G.nodes)
print(G.edges)
df=nx.to_pandas_edgelist(G)
print(df)
df=nx.to_pandas_edgelist(G,nodelist=['A'])
print(df)
A=nx.adjacency_matrix(G)
print(A.todense())

"""# Sudoku Graph"""

G=nx.sudoku_graph(3)
print(G.nodes)
print(G.edges)
nx.draw(G)
A=nx.adjacency_matrix(G)
print(A.todense())

G=nx.sudoku_graph(2)
print(G.nodes)
print(G.edges)
nx.draw(G)
A=nx.adjacency_matrix(G)
print(A.todense())

"""# Grid Graph"""

G=nx.grid_graph([2,3,4])
print(G.nodes)
print(G.edges)
A=nx.adjacency_matrix(G)
print(A.todense)
nx.draw(G)

G=nx.grid_graph([2,3])
print(G.nodes)
print(G.edges)
A=nx.adjacency_matrix(G)
print(A.todense)
nx.draw(G)

G=nx.grid_graph([2])
print(G.nodes)
print(G.edges)
A=nx.adjacency_matrix(G)
print(A.todense)
nx.draw(G)

"""# Introducing the PYG"""

import numpy as np

"""# Tensor"""

x=np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(x)
print(x.ndim)

"""# Downloading PYG"""

! python -c "import torch; print(torch.version.cuda)"

! python -c "import torch; print(torch.__version__)"

!pip install torch-geometric
!pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

import torch
from torch_geometric.data import Data

"""# DataSets"""

from torch_geometric.datasets import TUDataset
datasets=TUDataset(root='/temp/ENZYMES',name='ENZYMES')
print(datasets.num_classes)
print(len(datasets))
print(datasets.num_node_features)
print(datasets[0])

"""Data Nodes = ['Number of the Nodes','Node features']

Data Edges Index=['2','Number of the edges']

Data Edge features=['Number of the edges','one edge features']

Data y= [label of one data point]

# Plotting the Data Points
"""

data=datasets[0];
from torch_geometric.utils import to_networkx
G=to_networkx(data)
print(type(G))
print(type(data))
nx.draw(G)

"""# Cora Dataset"""

from torch_geometric.datasets import Planetoid
Cora_datasets=Planetoid(root='/temp/Cora',name='Cora')
print(len(Cora_datasets))
print(Cora_datasets.num_classes)
print(Cora_datasets.num_node_features)

cds=Cora_datasets[0]
print(cds.num_nodes)
print(cds.num_edges)
from torch_geometric.utils import to_networkx
G=to_networkx(x)
nx.draw(G)
print(G.number_of_nodes())
print(G.number_of_edges())
# to big to see

nx.write_graphml(G,'G_ex.graphml')
nx.write_gexf(G,'G_ex.gexf')
print(cds.x.shape)
print(cds.is_directed())

cds.x[:1].shape
cds.x[:5].shape

cds

cds

print(cds.train_mask.sum().item())
print(cds.test_mask.sum().item())
print(cds.val_mask.sum().item())

print(cds.x[0].shape)
type(cds.x[0])
