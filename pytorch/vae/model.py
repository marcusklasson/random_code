import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    
    def __init__(self, n_latents):
        super(VAE, self).__init__()
        self.encoder = Encoder(n_latents)
        self.decoder = Decoder(n_latents)
        self.n_latents = n_latents
        
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
        
class Encoder(nn.Module):
    
    def __init__(self, n_latents):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 2*n_latents)
        self.n_latents = n_latents
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        return out[:, :self.n_latents], out[:, self.n_latents:]
    
class Decoder(nn.Module):
    
    def __init__(self, n_latents):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(n_latents, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 784)
        self.n_latents = n_latents
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        return out # No sigmoid here, see loss