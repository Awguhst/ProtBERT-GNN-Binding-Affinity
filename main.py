import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from binding_model.model import HybridGNN
from data_preprocessing.data import load_data, ProteinLigandDataset
from binding_model.train_eval import train, val, evaluate

# Hyperparameters
random_seed = 66
hidden_channels = 256
descriptor_size = 1024
dropout_rate = 0.4
batch_size = 32
epochs = 50
lr = 0.0003
train_ratio = 0.90

# Initialize device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load protein model and tokenizer
protein_tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd')
protein_model = BertModel.from_pretrained('Rostlab/prot_bert_bfd').to(device)

# Load dataset
protein_embeddings, ligand_graphs, labels = load_data("data/Protein-ligand-binding.csv", protein_tokenizer, protein_model, device)
dataset = ProteinLigandDataset(protein_embeddings, ligand_graphs, labels)

# Split dataset
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

# Set a random seed for reproducibility
generator = torch.Generator().manual_seed(random_seed)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = HybridGNN(hidden_channels, descriptor_size, dropout_rate).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=5, gamma=0.75)

# Training loop
# Loop over each epoch for training
for epoch in range(epochs):
    # Train the model and get the training RMSE
    train_rmse = train(train_loader, model, criterion, optimizer, device)

    # Adjust learning rate
    scheduler.step()

    # Validate the model and get the validation RMSE
    val_rmse = val(val_loader, model, criterion, device)

    # Print progress
    print(f'Epoch: {epoch+1}/{epochs} | Train RMSE: {train_rmse:.4f}, '
          f'Validation RMSE: {val_rmse:.4f}')
    
evaluate(val_loader, model, device)