import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, embedding_dim,bottleneck_dim,global_latent_dim,beh_private_dim,neural_private_dim):
        super(Decoder, self).__init__()

        self.neural_encoder=nn.Linear(embedding_dim,bottleneck_dim)
        self.beh_encoder=nn.Linear(embedding_dim,bottleneck_dim)
        self.fusion=nn.Linear(bottleneck_dim*2,global_latent_dim)
        self.behprivate=nn.Linear(bottleneck_dim,beh_private_dim)
        self.neuralprivate=nn.Linear(bottleneck_dim,neural_private_dim)
        
    def forward(self, neural_represent,beh_represent):

        x=self.neural_encoder(neural_represent)
        y=self.beh_encoder(beh_represent)
        beh_private=self.behprivate(y)
        neural_private=self.neuralprivate(x)
        
        latent_cat=torch.cat((x,y),axis=-1)
        global_latent=self.fusion(latent_cat)

        beh_final=torch.cat((beh_private,global_latent),axis=-1)
        neural_final=torch.cat((neural_private,global_latent),axis=-1)
  
        return global_latent,beh_private,neural_private,beh_final,neural_final