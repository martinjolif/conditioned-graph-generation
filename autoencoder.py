import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, ChebConv, GATConv
from torch_geometric.nn import global_add_pool

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


class ImprovedDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(ImprovedDecoder, self).__init__()
        self.n_nodes = n_nodes
        self.latent_dim = latent_dim

        # Wider network for better expressivity
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            *[nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(n_layers - 1)],
            nn.Linear(hidden_dim * 2, n_nodes * n_nodes)
        )

        # Learnable initial temperature
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        batch_size = x.size(0)

        # Get logits
        logits = self.mlp(x)
        logits = logits.view(batch_size, self.n_nodes, self.n_nodes)

        # Ensure symmetry
        logits = (logits + logits.transpose(1, 2)) / 2

        # Apply sigmoid with temperature
        probs = torch.sigmoid(logits / torch.clamp(self.temperature, min=0.1))

        # During training, use straight-through estimator with threshold at 0.5
        if self.training:
            adj_hard = (probs > 0.5).float()
            adj = adj_hard.detach() - probs.detach() + probs
        else:
            adj = (probs > 0.5).float()

        # Zero out diagonal
        adj = adj * (1 - torch.eye(self.n_nodes, device=x.device))

        return adj

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out

class ChebEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2, K=3):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        self.convs.append(ChebConv(input_dim, hidden_dim, K=K))
        for layer in range(n_layers - 1):
            self.convs.append(ChebConv(hidden_dim, hidden_dim, K=K))

        self.bn = nn.BatchNorm1d(hidden_dim + 7)  #7 conditioning vector size
        self.fc = nn.Linear(hidden_dim + 7, latent_dim)   #7 conditioning vector size

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x
        cond = data.stats

        # Apply each ChebConv layer with ReLU and dropout
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.leaky_relu(x)
            if self.training:
                x = self.dropout(x)
        out = global_add_pool(x, data.batch)
        #ajoutez ca pour prendre en compte le conditionnement
        out = torch.cat([out, cond], dim=1)

        out = self.bn(out)
        out = self.fc(out)

        return out

#training much slower
class GATEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2, heads=4):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        # Initialize the first GAT layer with multi-head attention
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
        for layer in range(n_layers - 1):
            # Additional GAT layers; in each subsequent layer, input_dim becomes hidden_dim * heads
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))

        # Batch normalization and fully connected layer with conditioning
        self.bn = nn.BatchNorm1d(hidden_dim * heads + 7)  # 7 is the conditioning vector size
        self.fc = nn.Linear(hidden_dim * heads + 7, latent_dim)  # Latent dimension output

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x
        cond = data.stats

        # Apply each GATConv layer with LeakyReLU and dropout
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.leaky_relu(x)
            if self.training:
                x = self.dropout(x)

        # Global pooling and conditioning concatenation
        out = global_add_pool(x, data.batch)
        out = torch.cat([out, cond], dim=1)

        out = self.bn(out)
        out = self.fc(out)

        return out



# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        #self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.encoder = GATEncoder(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        #self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)
        self.decoder = ImprovedDecoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj

    def loss_function(self, data, beta=0.05):
        x_g  = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        
        #recon = F.l1_loss(adj, data.A, reduction='mean')
        recon = F.binary_cross_entropy(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld
        accuracy = ((adj > 0.5).float() == data.A).float().mean()
        print("accuracy edge prediction", accuracy)

        return loss, recon, kld
