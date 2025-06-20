# -*- coding: utf-8 -*-
"""GNN Final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r94J0Aa1XPz4f4aCCMjQ3ZC5V8N6Z13L
"""

import pandas as pd
DATA_PATH="/content/HIV.csv"
data=pd.read_csv(DATA_PATH)

data.head()

!pip install rdkit
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
sample_smiles=data["smiles"][4:30].values
sample_mols=[Chem.MolFromSmiles(smiles) for \
             smiles in sample_smiles]
grid=Draw.MolsToGridImage(sample_mols,molsPerRow=4,subImgSize=(300,300))
grid

import torch
!pip install torch-geometric
from torch_geometric.data import Dataset,Data
import numpy as np
import os
from rdkit.Chem import rdmolops
from tqdm import tqdm

import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset
import numpy as np
import os
from tqdm import tqdm
!pip install deepchem
import deepchem as dc
from rdkit import Chem

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]


    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row["smiles"])
            f = featurizer._featurize(mol)
            data = f.to_pyg_graph()
            data.y = self._get_label(row["HIV_active"])
            data.smiles = row["smiles"]
            if self.test:
                torch.save(data,
                    os.path.join(self.processed_dir,
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data,
                    os.path.join(self.processed_dir,
                                 f'data_{index}.pt'))


    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                 f'data_{idx}.pt'))
        return data

# Instantiate the dataset
dataset = MoleculeDataset(root="/content/data/",filename="HIV_train_oversampled.csv")

# Print the first element in the dataset
print(dataset[0])

# Alternatively, if you want to print specific attributes of the data
data = dataset[0]
print("Node features:", data.x)
print("Edge features:", data.edge_attr)

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
!pip install torch_geometric
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap ,global_max_pool as gmp

torch.manual_seed(42)

class GNN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GNN, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"]

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # Transformation layer
        self.conv1 = TransformerConv(feature_size,
                                    embedding_size,
                                    heads=n_heads,
                                    dropout=dropout_rate,
                                    edge_dim=edge_dim,
                                    beta=True)

        self.transf1 = Linear(embedding_size*n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        # Other layers
        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size,
                                                    embedding_size,
                                                    heads=n_heads,
                                                    dropout=dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))

            self.transf_layers.append(Linear(embedding_size*n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))


        # Linear layers
        self.linear1 = Linear(embedding_size*2, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons/2))
        self.linear3 = Linear(int(dense_neurons/2), 1)

    def forward(self, x, edge_attr, edge_index, batch_index):
        # Initial transformation
        edge_index = edge_index[:, ::2]
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            # Always aggregate last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x , edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                    )
                # Add current representation
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))

        x = sum(global_representation)

        # Output block
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)

        return x

import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,precision_score,recall_score,roc_auc_score
import numpy as np
from tqdm import tqdm
!pip install mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


mlflow.set_tracking_uri("file:///path/to/your/tracking/directory")

def count_parameters(model):
  return (sum(p.numel() for p in model.parameters() if p.requires_grad))

def train_one_epoch(epoch,model,train_loader,optimizer,loss_fn):
  all_preds=[]
  all_labels=[]
  running_loss=0.0
  step=0
  for _, batch in enumerate (tqdm(train_loader)):
    batch.to(device)
    optimizer.zero_grad()
    pred=model(batch.x.float(),batch.edge_attr.float(),batch.edge_index,batch.batch)
    loss=loss_fn(torch.squeeze(pred).float(),batch.y.float())
    loss.backward()
    optimizer.step()
    #Update the Tracking
    running_loss+=loss.item()
    step+=1

    all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
    all_labels.append(batch.y.cpu().detach().numpy())
  all_preds = np.concatenate(all_preds).ravel()
  all_labels = np.concatenate(all_labels).ravel()
  calculate_metrics(all_preds,all_labels,epoch,"train")
  return running_loss/step

def test(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        batch.to(device)
        pred = model(batch.x.float(),
                        batch.edge_attr.float(),
                        batch.edge_index,
                        batch.batch)
        loss = loss_fn(torch.squeeze(pred.float()), batch.y.float())
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    print(all_preds_raw[0][:10])
    print(all_preds[:10])
    print(all_labels[:10])
    calculate_metrics(all_preds, all_labels, epoch, "test")
    return running_loss/step


def calculate_metrics(y_pred,y_true,epoch,type):
  print("Confusion Matrix")
  print(confusion_matrix(y_pred,y_true))
  print(f1_score(y_true,y_pred))
  print(accuracy_score(y_true,y_pred))
  prec=precision_score(y_true,y_pred)
  rec=recall_score(y_true,y_pred)
  print(precision_score(y_true,y_pred))
  print(recall_score(y_true,y_pred))
  mlflow.log_metric(key=f"Precision_Score",value=float(prec),step=epoch)
  mlflow.log_metric(key=f"Recall_score",value=float(rec),step=epoch)
  try:
    roc=roc_auc_score(y_true,y_pred)
    print(roc_auc_score(y_true,y_pred))
    mlflow.log_metric(key=f"roc_auc_score",value=float(roc),step=epoch)
  except:
    mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
    print("Not defined roc")



def run_one_training(params):
  params=params[0]
  with mlflow.start_run() as run:
    for key in params.keys():
      mlflow.log_param(key,params[key])
    print("Loading the dataset")
    train_dataset = MoleculeDataset(root="data/", filename="HIV_train_oversampled.csv")
    test_dataset = MoleculeDataset(root="data/", filename="HIV_test.csv", test=True)
    params["model_edge_dim"]=train_dataset[0].edge_attr.shape[1]

    train_loader=DataLoader(train_dataset,batch_size=params["batch_size"],shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=params["batch_size"],shuffle=True)

    print("Loading Model")
    model_params={k: v for k , v in params.items() if k.startswith("model_")}
    model=GNN(feature_size=train_dataset[0].x.shape[1],model_params=model_params)
    model=model.to(device)
    print("number of paramters")
    print(count_parameters(model))
    mlflow.log_param("num_params",count_parameters(model))

    weight=torch.tensor([params["pos_weight"]], dtype=torch.float32). to(device)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer=torch.optim.SGD(model.parameters(),lr=params["learning_rate"],momentum=params["sgd_momentum"],weight_decay=params["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=params["scheduler_gamma"])

    #Start Training

    best_loss=10000
    early_stopping_counter=0
    for epoch in range (10):
      if early_stopping_counter<=10:
        model.train()
        loss=train_one_epoch(epoch,model,train_loader,optimizer,loss_fn)
        print(epoch)
        print(loss)
        mlflow.log_metric(key="Train Loss",value=float(loss),step=epoch)
        model.eval()
        if epoch%5==0:
          test_loss=test(epoch,model,test_loader,loss_fn)
          print(epoch)
          print(test_loss)
          mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)
          if float(loss)<best_loss:
            best_loss=loss
          else:
            early_stopping_counter+=1
        scheduler.step()
      else:
        print("early stopping no impovement")
        return (best_loss)

    return best_loss

# Define your parameters
params = {
    "batch_size": 128,
    "learning_rate": 0.01,
    "weight_decay": 0.0001,
    "sgd_momentum": 0.8,
    "scheduler_gamma": 0.8,
    "pos_weight": 1.3,
    "model_embedding_size": 64,
    "model_attention_heads": 3,
    "model_layers": 4,
    "model_dropout_rate": 0.2,
    "model_top_k_ratio": 0.5,
    "model_top_k_every_n": 1,
    "model_dense_neurons": 256
    # Add other parameters as needed
}

# Call the function
run_one_training([params])
