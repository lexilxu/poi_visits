import sys
import os
import torch
import numpy as np
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import gtnet
from util import StandardScaler, DataLoaderM


class SimpleScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return data * self.std + self.mean


def load_dataset_subset(data_dir, batch_size, num_nodes=20, seq_in_len=14, seq_out_len=7):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, category + '.npz'))
        x = cat_data['x'][:, :, :num_nodes, :]  # (S, C, N, T)
        y = cat_data['y'][:, :, :num_nodes, 0]  # (S, T, N) - remove last dim
        data['x_' + category] = x
        data['y_' + category] = y
        print(f"  {category}: x={x.shape}, y={y.shape}")
    
    scaler = SimpleScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    
    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], batch_size)
    data['scaler'] = scaler
    return data


def load_adj_subset(adj_pkl, num_nodes=20):
    with open(adj_pkl, 'rb') as f:
        adj_tuple = pickle.load(f)
    # adj_tuple is (sensor_ids, sensor_id_to_ind, adj)
    full_adj = adj_tuple[2]
    adj_subset = full_adj[:num_nodes, :num_nodes]
    adj_subset = adj_subset - np.eye(num_nodes)
    return adj_subset


class SimpleTrainer:
    def __init__(self, model, lrate, wdecay, clip, seq_out_len, scaler, device):
        self.model = model
        self.model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss_fn = torch.nn.MSELoss()
        self.clip = clip
        self.seq_out_len = seq_out_len
        self.scaler = scaler
        self.device = device
    
    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = output
        loss = self.loss_fn(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.item(), 0.0, 0.0
    
    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
            output = output.transpose(1, 3)
            real = torch.unsqueeze(real_val, dim=1)
            predict = output
            loss = self.loss_fn(predict, real)
        return loss.item(), 0.0, 0.0


def main():
    device = torch.device('cpu')
    
    data_dir = "F:/博一/LLM-POI/poi_visits/data/splits/mtgnn"
    adj_pkl = "F:/博一/LLM-POI/poi_visits/data/adjacency/static/mtgnn_adj_identity.pkl"
    num_nodes = 20
    seq_in_len = 14
    seq_out_len = 7
    batch_size = 4
    epochs = 3
    
    print("=" * 60)
    print("MTGNN CPU Training Test")
    print("=" * 60)
    
    print("\nLoading data (subset with num_nodes=20)...")
    dataloader = load_dataset_subset(data_dir, batch_size, num_nodes=num_nodes)
    detected_in_dim = dataloader['x_train'].shape[1]
    print(f"\nDetected in_dim: {detected_in_dim}")
    
    print("\nLoading adjacency matrix (subset)...")
    predefined_A = load_adj_subset(adj_pkl, num_nodes)
    predefined_A = torch.tensor(predefined_A, dtype=torch.float32).to(device)
    print(f"  Adjacency shape: {predefined_A.shape}")
    
    print("\nInitializing model...")
    model = gtnet(
        gcn_true=True,
        buildA_true=False,
        gcn_depth=2,
        num_nodes=num_nodes,
        device=device,
        predefined_A=predefined_A,
        static_feat=None,
        dropout=0.3,
        subgraph_size=10,
        node_dim=20,
        dilation_exponential=1,
        conv_channels=16,
        residual_channels=16,
        skip_channels=32,
        end_channels=64,
        seq_length=seq_in_len,
        in_dim=detected_in_dim,
        out_dim=seq_out_len,
        layers=2,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True,
        cat_vocab_size=0,
        cat_emb_dim=8,
        naics2_vocab_size=0,
        naics2_emb_dim=8,
        naics4_vocab_size=0,
        naics4_emb_dim=8,
        brand_vocab_size=0,
        brand_emb_dim=8,
        parent_vocab_size=0,
        parent_emb_dim=8
    ).to(device)
    
    n_params = sum(p.nelement() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    print(f"  Receptive field: {model.receptive_field}")
    
    print("\nCreating trainer...")
    engine = SimpleTrainer(model, 0.001, 0.0001, 5, seq_out_len, dataloader['scaler'], device)
    
    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = []
        
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).transpose(1, 2).to(device)
            
            metrics = engine.train(trainx, trainy)
            epoch_loss.append(metrics[0])
        
        avg_loss = np.mean(epoch_loss)
        print(f"  Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.4f}")
    
    print("\nValidation test...")
    model.eval()
    val_losses = []
    with torch.no_grad():
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).transpose(1, 2).to(device)
            metrics = engine.eval(testx, testy)
            val_losses.append(metrics[0])
    
    print(f"  Avg Val Loss: {np.mean(val_losses):.4f}")
    
    print("\n" + "=" * 60)
    print("CPU Training Test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
