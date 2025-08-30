import torch
import torch.nn as nn
import torch.nn.init as init

C0 = 0.28209479177387814

class RegionConcealing(nn.Module):
    def __init__(self, D=4, W=256, input_ch=3, input_ch_time=3, args=None):
        super(RegionConcealing, self).__init__() 
        
        # naive version
        self.args = args
        # self.alpha = nn.Parameter(torch.tensor(0.5).cuda(), requires_grad=True)
        # self.beta = nn.Parameter(torch.tensor(0.5).cuda(), requires_grad=True)
        # self.gamma = nn.Parameter(torch.tensor(0.5).cuda(), requires_grad=True)
        # self.net_activation = nn.Sigmoid()
        
        # hybrid
        # self.alpha = nn.Conv1d(6, 1, kernel_size=7, padding=3)
        # self.beta = nn.Conv1d(6, 1, kernel_size=7, padding=3)
        # self.gamma = nn.Parameter(torch.tensor(0.5).cuda(), requires_grad=True)
        # self.net_activation = nn.Sigmoid()
        
        # wild
        feat_in = 3
        hiddden_size = 128
        
        self.under_mlp = nn.Sequential(
            nn.Linear(args.illumination_embedding_dim + feat_in, hiddden_size),
            nn.Sigmoid(),
            nn.Linear(hiddden_size, hiddden_size),
            nn.Sigmoid(),
            nn.Linear(hiddden_size, 2),
        )
        
        self.over_mlp = nn.Sequential(
            nn.Linear(args.illumination_embedding_dim + feat_in, hiddden_size),
            nn.Sigmoid(),
            nn.Linear(hiddden_size, hiddden_size),
            nn.Sigmoid(),
            nn.Linear(hiddden_size, 2),
        )
        
        
        # feat_in = 3
        # hiddden_size = 128
        
        # self.under_mlp = nn.Sequential(
        #     nn.Linear(args.illumination_embedding_dim, hiddden_size),
        #     nn.Sigmoid(),
        #     nn.Linear(hiddden_size, hiddden_size),
        #     nn.Sigmoid(),
        #     nn.Linear(hiddden_size, 2),
        # )
        
        # self.over_mlp = nn.Sequential(
        #     nn.Linear(args.illumination_embedding_dim, hiddden_size),
        #     nn.Sigmoid(),
        #     nn.Linear(hiddden_size, hiddden_size),
        #     nn.Sigmoid(),
        #     nn.Linear(hiddden_size, 2),
        # )
        
        for name, param in self.under_mlp.named_parameters():
            if 'weight' in name:
                # init.ones_(param)
                init.uniform_(param, a=0.02, b=0.2)
                # init.xavier_uniform_(param)
            if 'bias' in name: 
                init.constant_(param, val=0)
        
        for name, param in self.over_mlp.named_parameters():
            if 'weight' in name:
                # init.ones_(param)
                init.uniform_(param, a=0.02, b=0.2)
                # init.xavier_uniform_(param)
            if 'bias' in name: 
                init.constant_(param, val=0)
        
    def forward(self, x, embedding, ill_type):
        
        # embedding
        if ill_type == 'low_light':
            mlp = self.under_mlp
        else:
            mlp = self.over_mlp
        offset, mul = torch.split(mlp(torch.cat((x, embedding), dim=-1))*0.01, [1, 1], dim=-1)
        # offset, mul = torch.split(mlp(embedding)*0.01, [1, 1], dim=-1)
        # print(offset.mean().item(), mul.mean().item())
        out = (x * mul + offset / C0)
        # print(out.min(), out.max())
        return out