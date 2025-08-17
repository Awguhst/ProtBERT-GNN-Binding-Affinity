import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import from_smiles
from torch_geometric.data import Data

# Dataset class for handling protein-ligand pairs
class ProteinLigandDataset(Dataset):
    def __init__(self, protein_embeddings, ligand_graphs, labels):
        self.protein_embeddings = protein_embeddings
        self.ligand_graphs = ligand_graphs
        self.labels = labels

    def __len__(self):
        return len(self.protein_embeddings)

    def __getitem__(self, idx):
        protein_embedding = self.protein_embeddings[idx]
        ligand_graph = self.ligand_graphs[idx]
        label = self.labels[idx]
        return protein_embedding, ligand_graph, label

# Function to get protein embedding using ProtBERT
def get_protein_embedding(protein_sequence, protein_tokenizer, protein_model, device):
    
    inputs = protein_tokenizer(protein_sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = protein_model(**inputs)
        embeddings = outputs.last_hidden_state
    pooled_embedding = embeddings.mean(dim=1)  # Mean pooling
    return pooled_embedding

# Function to generate ligand graph from SMILES
def generate_ligand_graph(smiles):
    g = from_smiles(smiles)
    g.x = g.x.float()  # Ensure node features are float
    return g

def load_data(file_path, protein_tokenizer, protein_model, device):
    df = pd.read_csv(file_path)
    df = df.sample(n=10000, random_state=66)
    df.reset_index(drop=True, inplace=True)
    df = df.reset_index(drop=True)

    # Prepare the protein embeddings and ligand graphs
    protein_embeddings = []
    ligand_graphs = []

    for i, row in df.iterrows():
        # Protein sequence to embedding
        protein_embeddings.append(get_protein_embedding(row['protein_sequence'], protein_tokenizer, protein_model, device))
        
        # SMILES to graph
        ligand_graph = generate_ligand_graph(row['compound_smiles'])
        ligand_graphs.append(ligand_graph)
        
        # Print progress every 500 samples
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} samples...")

    # Convert protein embeddings to tensor and move to the device
    protein_embeddings = torch.cat(protein_embeddings, dim=0).to(device)

    # Labels for affinity prediction
    labels = torch.tensor(df['label'].values, dtype=torch.float).view(-1, 1).to(device)
    
    return protein_embeddings, ligand_graphs, labels
