import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import random
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import AllChem

from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool
import torch.optim as optim

from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

ablation_study_id = 5

def clean_name(name):
    return re.sub(r'[^A-Za-z0-9]', '', name).upper()


def compute_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * n_bits
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp)

def one_hot_encoding(x, allowable):
    return [int(x == s) for s in allowable]

def atom_features(atom):
    features = []
    features += one_hot_encoding(atom.GetSymbol(), ATOM_LIST)
    features += one_hot_encoding(atom.GetDegree(), DEGREE_LIST)
    features += one_hot_encoding(atom.GetImplicitValence(), VALENCE_LIST)
    features.append(atom.GetFormalCharge())
    features.append(atom.GetNumRadicalElectrons())
    features += one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION_LIST)
    features.append(int(atom.GetIsAromatic()))
    features += one_hot_encoding(atom.GetTotalNumHs(), TOTAL_H_LIST)
    features.append(int(atom.IsInRing()))
    
    chirality = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else None
    features += one_hot_encoding(chirality, CHIRALITY_LIST)
    
    return features

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))  # undirected

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)


def modified_mgataf_collate_fn(batch):
    graphs, fingerprints, ccl_feats, gsva_feats, labels = zip(*batch)

    # Batch graph objects using PyG
    graph_batch = Batch.from_data_list(graphs)

    # Stack other tensors
    fingerprint_batch = torch.stack(fingerprints)
    ccl_feat_batch = torch.stack(ccl_feats)
    gsva_feat_batch = torch.stack(gsva_feats)
    label_batch = torch.stack(labels)

    return graph_batch, fingerprint_batch, ccl_feat_batch, gsva_feat_batch, label_batch


class ModifiedMGATAFDataset(Dataset):
    def __init__(self, gdsc_df, fingerprint_dict, cell_feature_matrix, gsva_matrix, graph_dict):
        self.df = gdsc_df
        self.fingerprint_dict = fingerprint_dict  # drug_id -> np.array or list
        self.cell_features = cell_feature_matrix  # DataFrame: index=cell_line_name, values=mutation+cnv
        self.gsva_matrix = gsva_matrix            # DataFrame: index=cell_line_name, values=gsva scores
        self.graphs = graph_dict                  # drug_id -> graph object (PyG or DGL)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        drug_id = row["DRUG_ID"]
        cell_line = row["CLEAN_CELL_LINE"]

        # Drug molecular graph
        graph_data = self.graphs[drug_id]

        # Fingerprint vector
        fingerprint = torch.tensor(self.fingerprint_dict[drug_id], dtype=torch.float)

        # Cell line mutation/CNV features
        ccl_feat = torch.tensor(self.cell_features.loc[cell_line].values, dtype=torch.float)

        # GSVA pathway scores
        if cell_line not in gsva_matrix.index:
            raise ValueError(f"{cell_line} not found in GSVA matrix.")
        gsva_feat = torch.tensor(self.gsva_matrix.loc[cell_line].values, dtype=torch.float)

        # Target IC50
        ic50 = torch.tensor([row["LN_IC50"]], dtype=torch.float)

        return graph_data, fingerprint, ccl_feat, gsva_feat, ic50


class DrugGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=128, num_layers=3, heads=4):
        super().__init__()
        self.gats = nn.ModuleList()
        self.num_layers = num_layers
        self.att_proj = nn.Linear(hidden_dim, 1)

        self.gats.append(GATConv(in_dim, hidden_dim, heads=heads, concat=False))
        for _ in range(1, num_layers):
            self.gats.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        layer_outputs = []

        for gat in self.gats:
            x = F.relu(gat(x, edge_index))
            pooled = global_add_pool(x, batch)
            layer_outputs.append(pooled)

        h = torch.stack(layer_outputs, dim=1)  # [B, L, D]
        attn_weights = F.softmax(self.att_proj(h).squeeze(-1), dim=1)  # [B, L]
        h_weighted = (h * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return h_weighted


class FingerprintEncoder(nn.Module):
    def __init__(self, in_dim=2048, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        # Dummy pass to infer output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_dim)
            conv_out = self.conv(dummy)
            conv_flat_dim = conv_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_flat_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.fc(x)
        return x


class CellLineEncoder(nn.Module):
    def __init__(self, in_dim=735, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        # Dynamically infer flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_dim)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 735]
        x = self.conv(x)
        x = self.fc(x)
        return x


class GSVAEncoder(nn.Module):
    def __init__(self, in_dim=658, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_dim)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 658]
        x = self.conv(x)
        x = self.fc(x)
        return x


class AdaptiveFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # x1: drug + fp, x2: ccl
        concat = torch.cat([x1, x2], dim=1)
        gate = self.gate_layer(concat)

        fused = gate * x1 + (1 - gate) * x2
        return fused


class IC50Predictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)


class ModifiedMGATAFModel(nn.Module):
    def __init__(self, atom_feat_dim=55, fingerprint_dim=2048, ccl_dim=735, gsva_dim=658, hidden_dim=128):
        super().__init__()
        self.drug_encoder = DrugGraphEncoder(in_dim=atom_feat_dim, hidden_dim=hidden_dim)
        # self.fp_encoder = FingerprintEncoder(in_dim=fingerprint_dim, out_dim=hidden_dim)

        # self.ccl_encoder = CellLineEncoder(in_dim=ccl_dim, out_dim=hidden_dim)
        self.gsva_encoder = GSVAEncoder(in_dim=gsva_dim, out_dim=hidden_dim)
        
        # self.drug_fusion = AdaptiveFusion(input_dim=hidden_dim, hidden_dim=hidden_dim)
        # self.cellline_fusion = AdaptiveFusion(input_dim=hidden_dim, hidden_dim=hidden_dim)  # new!
        self.fusion = AdaptiveFusion(input_dim=hidden_dim, hidden_dim=hidden_dim)

        self.regressor = IC50Predictor(input_dim=hidden_dim)

    def forward(self, graph_data, fingerprint, ccl_feat, gsva_feat):
        # Drug branch
        drug_repr = self.drug_encoder(graph_data)
        # fp_repr = self.fp_encoder(fingerprint)
        # drug_combined = drug_repr + fp_repr
        # drug_combined = self.drug_fusion(drug_repr, fp_repr)

        # Cell line branch
        # ccl_repr = self.ccl_encoder(ccl_feat)
        gsva_repr = self.gsva_encoder(gsva_feat)
        # final_ccl_repr = self.cellline_fusion(ccl_repr, gsva_repr)

        # Final fusion
        fused = self.fusion(drug_repr, gsva_repr)

        return self.regressor(fused)


# Set seeds reproducibly
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Log results to a file
def log_results(file_path, log_str):
    with open(file_path, 'a') as f:
        f.write(log_str + '\n')


def evaluate_on_test_set(model, test_loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for graph_data, fingerprint, ccl_feat, gsva_feat, label in test_loader:
            graph_data = graph_data.to(device)
            fingerprint = fingerprint.to(device)
            ccl_feat = ccl_feat.to(device)
            gsva_feat = gsva_feat.to(device)
            label = label.to(device)

            preds = model(graph_data, fingerprint, ccl_feat, gsva_feat)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    # Flatten predictions and labels
    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_labels).flatten()

    # Compute metrics
    test_rmse = root_mean_squared_error(y_true, y_pred)
    test_pcc = pearsonr(y_true, y_pred)[0]

    print(f"✅ Test RMSE: {test_rmse:.4f}")
    print(f"✅ Test PCC:  {test_pcc:.4f}")

    return test_rmse, test_pcc



def train_and_evaluate(seed):
    set_seed(seed)

    print(f"Total {len(common_cell_lines)} common cell lines")
    # common_cell_lines = np.array(common_cell_lines)
    np.random.shuffle(common_cell_lines)
    cell_lines_train = common_cell_lines[: int(0.8*common_cell_lines.shape[0])]
    cell_lines_test = common_cell_lines[int(0.8*common_cell_lines.shape[0]) : int(0.9*common_cell_lines.shape[0])]
    cell_lines_val = common_cell_lines[int(0.9*common_cell_lines.shape[0]):]
    print(f"Train: {cell_lines_train.shape[0]} cell lines")
    print(f"Test: {cell_lines_test.shape[0]} cell lines")
    print(f"Val: {cell_lines_val.shape[0]} cell lines")

    train_set = ModifiedMGATAFDataset(
        gdsc_df=gdsc_df[gdsc_df["CLEAN_CELL_LINE"].isin(cell_lines_train)].reset_index(drop=True),
        fingerprint_dict=fingerprint_dict,
        cell_feature_matrix=binary_feature_matrix,
        gsva_matrix=gsva_matrix,
        graph_dict=precomputed_graphs
    )
    val_set = ModifiedMGATAFDataset(
        gdsc_df=gdsc_df[gdsc_df["CLEAN_CELL_LINE"].isin(cell_lines_val)].reset_index(drop=True),
        fingerprint_dict=fingerprint_dict,
        cell_feature_matrix=binary_feature_matrix,
        gsva_matrix=gsva_matrix,
        graph_dict=precomputed_graphs
    )
    test_set = ModifiedMGATAFDataset(
        gdsc_df=gdsc_df[gdsc_df["CLEAN_CELL_LINE"].isin(cell_lines_test)].reset_index(drop=True),
        fingerprint_dict=fingerprint_dict,
        cell_feature_matrix=binary_feature_matrix,
        gsva_matrix=gsva_matrix,
        graph_dict=precomputed_graphs
    )

    print(f"Train: {len(train_set)} pairs")
    print(f"Val: {len(val_set)} pairs")
    print(f"Val: {len(test_set)} pairs")

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=modified_mgataf_collate_fn)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, collate_fn=modified_mgataf_collate_fn)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=modified_mgataf_collate_fn)


    model = ModifiedMGATAFModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    num_epochs = 100
    warmup_epochs = 10
    t_max = num_epochs - warmup_epochs

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # Trackers
    train_rmse_list = []
    val_rmse_list = []
    val_pcc_list = []


    # Paths
    checkpoint_model_path = f"models/checkpoints/modified_mgataf_ablation/modified_mgataf_checkpoint_model_{ablation_study_id}.pt"
    checkpoint_opt_path = f"models/checkpoints/modified_mgataf_ablation/mgataf_checkpoint_optim_{ablation_study_id}.pt"
    checkpoint_meta_path = f"models/checkpoints/modified_mgataf_ablation/mgataf_checkpoint_meta_{ablation_study_id}.pt"

    model_save_path = f"models/modified_mgataf_type{ablation_study_id}_best_model"

    # Defaults
    start_epoch = 0

    # Resume
    # if os.path.exists(checkpoint_model_path) and os.path.exists(checkpoint_opt_path) and os.path.exists(checkpoint_meta_path):
    #     print("🔄 Resuming model...")
    #     model.load_state_dict(torch.load(checkpoint_model_path))

    #     # if os.path.exists(checkpoint_opt_path):
    #     optimizer.load_state_dict(torch.load(checkpoint_opt_path))

    #     # if os.path.exists(checkpoint_meta_path):
    #     meta = torch.load(checkpoint_meta_path)
    #     start_epoch = meta["epoch"] + 1
    #     best_val_loss = meta["best_val_loss"]
    #     patience_counter = meta["patience_counter"]

    #     print(f"✅ Resumed from epoch {start_epoch} | Best Val Loss: {best_val_loss:.4f}")


    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_preds, train_labels = [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
        for batch_idx, (graph_data, fingerprint, ccl_feat, gsva_feat, label) in enumerate(loop):
            graph_data = graph_data.to(device)
            fingerprint = fingerprint.to(device)
            ccl_feat = ccl_feat.to(device)
            gsva_feat = gsva_feat.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(graph_data, fingerprint, ccl_feat, gsva_feat)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            train_preds.append(output.detach().cpu().numpy())
            train_labels.append(label.cpu().numpy())
            
            # # Every N batches, compute and show running metrics
            # if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == len(train_loader):
            #     pred_flat = np.concatenate(train_preds).flatten()
            #     label_flat = np.concatenate(train_labels).flatten()
            #     rmse = root_mean_squared_error(label_flat, pred_flat)
            #     try:
            #         pcc = pearsonr(label_flat, pred_flat)[0]
            #     except:
            #         pcc = float('nan')

            #     loop.set_postfix(train_rmse=f"{rmse:.4f}", train_pcc=f"{pcc:.4f}")

        train_preds_flat = np.concatenate(train_preds).flatten()
        train_labels_flat = np.concatenate(train_labels).flatten()
        train_rmse = root_mean_squared_error(train_labels_flat, train_preds_flat)
        train_rmse_list.append(train_rmse)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for graph_data, fingerprint, ccl_feat, gsva_feat, label in val_loader:
                graph_data = graph_data.to(device)
                fingerprint = fingerprint.to(device)
                ccl_feat = ccl_feat.to(device)
                gsva_feat = gsva_feat.to(device)
                label = label.to(device)

                output = model(graph_data, fingerprint, ccl_feat, gsva_feat)
                val_preds.append(output.cpu().numpy())
                val_labels.append(label.cpu().numpy())

        val_preds_flat = np.concatenate(val_preds).flatten()
        val_labels_flat = np.concatenate(val_labels).flatten()
        val_rmse = root_mean_squared_error(val_labels_flat, val_preds_flat)
        val_pcc = pearsonr(val_preds_flat, val_labels_flat)[0]

        val_rmse_list.append(val_rmse)
        val_pcc_list.append(val_pcc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch+1}] Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Val PCC: {val_pcc:.4f} | LR = {current_lr:.6f}")

        scheduler.step()

        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path + f"_seed_{seed}" + ".pt")
            print(f"[Saved Best Model] | Val RMSE: {val_rmse:.4f} | Val PCC: {val_pcc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print("Early stopping triggered.")
                break

        # if (epoch % 5) == 0:
        #     torch.save(model.state_dict(), checkpoint_model_path)
        #     torch.save(optimizer.state_dict(), checkpoint_opt_path)
        #     torch.save({
        #         "epoch": epoch,
        #         "best_val_loss": best_val_loss,
        #         "patience_counter": patience_counter
        #     }, checkpoint_meta_path)
        #     print(f"[Checkpointed Model and State] | Val RMSE: {val_rmse:.4f} | Val PCC: {val_pcc:.4f}")


    # After training: Evaluate on test set
    model.load_state_dict(torch.load(model_save_path + f"_seed_{seed}" + ".pt"))
    test_rmse, test_pcc = evaluate_on_test_set(model, test_loader)  # use your actual eval function
    return test_rmse, test_pcc  


torch.cuda.empty_cache()

gdsc_df = pd.read_csv("dataset/GDSC_SMILES_merged.csv", index_col=0)
ccl_rep_df = pd.read_csv("dataset/PANCANCER_Genetic_feature.csv")
gsva_df = pd.read_csv("dataset/ccle_gsva_scores.csv", index_col=0)


# Pivot to get binary matrix: rows = cell lines, columns = features
binary_feature_matrix = ccl_rep_df.pivot_table(
    index="cell_line_name",
    columns="genetic_feature",
    values="is_mutated",
    fill_value=0
).astype(int)

binary_feature_matrix.columns = binary_feature_matrix.columns.astype(str)
binary_feature_matrix.index = binary_feature_matrix.index.str.upper()

gdsc_df["CLEAN_CELL_LINE"] = gdsc_df["CELL_LINE_NAME"].apply(clean_name)
unique_drugs = gdsc_df[["DRUG_ID", "SMILES"]].drop_duplicates().reset_index(drop=True)
unique_drugs["FINGERPRINT"] = unique_drugs["SMILES"].apply(compute_morgan)
fingerprint_dict = dict(zip(unique_drugs["DRUG_ID"], unique_drugs["FINGERPRINT"]))

gdsc_df["CELL_LINE_NAME"] = gdsc_df["CELL_LINE_NAME"].str.strip().str.upper()
cell_lines_obs = set(gdsc_df["CELL_LINE_NAME"].unique())

gsva_df.columns = gsva_df.columns.str.strip().str.upper()
cell_lines_available = sorted(set(gsva_df.columns.str.split("_").str[0].str.upper()))
cell_lines_available = {clean_name(name) for name in cell_lines_available}
cell_lines_obs = {clean_name(name) for name in cell_lines_obs}
common_cell_lines = cell_lines_obs.intersection(cell_lines_available)
print("Now common cell lines:", len(common_cell_lines))

gsva_df.columns = gsva_df.columns.str.split("_").str[0].str.upper().to_series().apply(clean_name)
gsva_df = gsva_df.loc[:, ~gsva_df.columns.duplicated()]


# Define categorical vocabularies
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'B', 'Si', 'Na', 'K', 'Li', 'Mg', 'Ca', 'Fe', 'Zn', 'Se', 'Cu']
DEGREE_LIST = list(range(0, 11))
VALENCE_LIST = list(range(0, 7))
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
TOTAL_H_LIST = [0, 1, 2, 3, 4]
CHIRALITY_LIST = ['R', 'S']


# Step 1: Clean the GDSC cell line column
# gdsc_df["CELL_LINE_NAME_CLEAN"] = gdsc_df["CELL_LINE_NAME"].apply(clean_name)

# Step 2: Clean the index of the binary feature matrix
binary_feature_matrix.index = binary_feature_matrix.index.to_series().apply(clean_name)
binary_feature_matrix = binary_feature_matrix[~binary_feature_matrix.index.duplicated(keep='first')]

# Step 3: Clean GSVA matrix columns
gsva_df.columns = gsva_df.columns.to_series().apply(clean_name)
gsva_df = gsva_df.loc[:, ~gsva_df.columns.duplicated(keep='first')]

# Step 4: Determine valid IDs
valid_drugs = set(fingerprint_dict.keys())
valid_mutcnv_cells = set(binary_feature_matrix.index)
valid_gsva_cells = set(gsva_df.columns)

# Step 5: Get cell lines common to all three
common_cell_lines = valid_mutcnv_cells & valid_gsva_cells & set(gdsc_df["CLEAN_CELL_LINE"])
common_cell_lines = np.array(list(common_cell_lines))
# Step 6: Filter gdsc_df to keep only rows with common cell lines and valid drugs/SMILES
gdsc_df = gdsc_df[
    gdsc_df["DRUG_ID"].isin(valid_drugs) &
    gdsc_df["CLEAN_CELL_LINE"].isin(common_cell_lines) &
    gdsc_df["SMILES"].notna()
].reset_index(drop=True)

# normalise the LN_IC50 values
gdsc_df['LN_IC50'] = 1 / (np.exp(-0.1 * gdsc_df['LN_IC50']) + 1)

# Step 7: Filter binary_feature_matrix and gsva_matrix to keep only common cell lines
binary_feature_matrix = binary_feature_matrix.loc[common_cell_lines]
gsva_df = gsva_df.loc[:, list(common_cell_lines)]  # since cell lines are columns

drug_smiles = gdsc_df.drop_duplicates(subset="DRUG_ID")[["DRUG_ID", "SMILES"]]

precomputed_graphs = {}
for _, row in drug_smiles.iterrows():
    drug_id = row["DRUG_ID"]
    smi = row["SMILES"]
    graph = smiles_to_graph(smi)
    if graph is not None:
        precomputed_graphs[drug_id] = graph

gsva_matrix = gsva_df.T  # Now cell lines are rows


# dataset = ModifiedMGATAFDataset(
#     gdsc_df=gdsc_df,
#     fingerprint_dict=fingerprint_dict,
#     cell_feature_matrix=binary_feature_matrix,
#     gsva_matrix=gsva_matrix,
#     graph_dict=precomputed_graphs
# )

# sample = dataset[0]
# graph, fingerprint, cell_feat, gsva_feat, label = sample

# print("Graph:")
# print(graph)
# print("Graph node features shape:", graph.x.shape)
# print("Graph edge_index shape:", graph.edge_index.shape)
# print()
# print("Fingerprint shape:", fingerprint.shape)
# print("Cell line features shape:", cell_feat.shape)
# print("GSVA features shape:", gsva_feat.shape)
# print("Label (ln_IC50):", label)

print("CUDA Available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main loop over seeds
output_file = f"fused_mgataf_type{ablation_study_id}_results_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
seeds = [42, 52, 62, 72, 82]
all_rmse, all_pcc = [], []

for seed in seeds:
    print(f"Running for seed {seed}")
    rmse, pcc = train_and_evaluate(seed)
    all_rmse.append(rmse)
    all_pcc.append(pcc)
    log_results(output_file, f"Seed {seed}: RMSE = {rmse:.4f}, PCC = {pcc:.4f}")

# Final summary stats
mean_rmse, std_rmse = np.mean(all_rmse), np.std(all_rmse)
mean_pcc, std_pcc = np.mean(all_pcc), np.std(all_pcc)

log_results(output_file, "\n==== Summary ====")
log_results(output_file, f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
log_results(output_file, f"PCC : {mean_pcc:.4f} ± {std_pcc:.4f}")
