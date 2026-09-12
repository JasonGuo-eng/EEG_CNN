"""Learn EEG filters, summarize log power, and predict age."""
import math
import torch
from torch import nn

class RawPowerCNN(nn.Module):
    """Learn spatial and temporal filters, then aggregate log power by subject/state."""
    def __init__(self,width=16,dropout=0.35):
        super().__init__()
        self.width = width
        self.spatial = nn.Conv1d(64,width,1,bias=False)
        nn.init.orthogonal_(self.spatial.weight.reshape(width,64))
        self.temporal = nn.Conv1d(width,width*8,101,padding=50,
                                  stride=4,groups=width,bias=False)
        # Physiologically meaningful initialization, with all filter weights learnable.
        t = torch.arange(101,dtype=torch.float32)/100.-0.5
        filters = []
        for _ in range(width):
            for freq in (2.5,5.5,9.,11.,14.,20.,27.,36.):
                wave = torch.cos(2*math.pi*freq*t)*torch.exp(-0.5*(t/0.15)**2)
                wave -= wave.mean()
                filters.append(wave/wave.norm())
        with torch.no_grad():
            self.temporal.weight.copy_(torch.stack(filters)[:,None,:])
        features = width*8+64
        self.feature_norm = nn.BatchNorm1d(features)
        self.head = nn.Sequential(nn.Linear(features*2,64),nn.ELU(),
                                  nn.Dropout(dropout),nn.Linear(64,1))

    def encode(self,x):
        x = x-x.mean(-1,keepdim=True)
        filtered = self.temporal(self.spatial(x))
        learned = torch.log(filtered.square().mean(-1).clamp_min(1e-8))
        sensor = torch.log(x.square().mean(-1).clamp_min(1e-8))
        return self.feature_norm(torch.cat([learned,sensor],dim=1))

    def forward(self,x,aux=None):
        # B, eyes closed/open, windows per state, channels, time
        b,s,k,c,t = x.shape
        z = self.encode(x.reshape(b*s*k,c,t)).reshape(b,s,k,-1).mean(2)
        return self.head(z.flatten(1)).squeeze(-1)

CONFIG = dict(kind="raw", width=16, dropout=0.35, loss="huber",
              lr=0.001, weight_decay=0.03, epochs=120, batch=16, windows=4)


def build_model(config=CONFIG):
    return RawPowerCNN(width=config["width"], dropout=config["dropout"])
