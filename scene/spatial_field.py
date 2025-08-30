import torch
import torch.nn as nn
import torch.nn.init as init


class SpatialConcealing(nn.Module):
    def __init__(self, D=4, W=256, input_ch=3, input_ch_time=3, args=None):
        super(SpatialConcealing, self).__init__() 
        
        # naive version
        self.args = args
        hiddden_size = 128
        
        self.spatial_mlp = nn.Sequential(
            nn.Linear(args.illumination_embedding_dim, hiddden_size),
            nn.Tanh(),
            nn.Linear(hiddden_size, hiddden_size),
            nn.Tanh(),
            nn.Linear(hiddden_size, 1),
        )
        
        # for name, param in self.spatial_mlp.named_parameters():
        #     if 'weight' in name:
        #         # init.ones_(param)
        #         init.uniform_(param, a=0.02, b=0.2)
        #         # init.xavier_uniform_(param)
        #     if 'bias' in name: 
        #         init.constant_(param, val=0)
        
    def forward(self, x, embedding):
        alpha = self.spatial_mlp(embedding)[None].permute(2,0,1)
        out = x + alpha*x*(1-x)
        return out