import torch
import torch.nn as nn
from .loss import *
from .acne import *
from .decoder import *

def evaluate_pose(x, att):
    # x: B3N, att: B1KN1
    # ts: B3k1
    pai = att.sum(dim=3, keepdim=True) # B1K11
    att = att / torch.clamp(pai, min=1e-3)
    ts = torch.sum(
        att * x[:, :, None, :, None], dim=3) # B3K1
    return ts

def spatial_variance(x, att, norm_type="l2"):
    pai = att.sum(dim=3, keepdim=True) # B1K11
    att = att / torch.clamp(pai, min=1e-3)
    ts = torch.sum(
        att * x[:, :, None, :, None], dim=3) # B3K1

    x_centered = x[:, :, None] - ts # B3KN
    x_centered = x_centered.permute(0, 2, 3, 1) # BKN3
    att = att.squeeze(1) # BKN1
    cov = torch.matmul(
        x_centered.transpose(3, 2), att * x_centered) # BK33
    
    # l2 norm
    vol = torch.diagonal(cov, dim1=-2, dim2=-1).sum(2) # BK
    if norm_type == "l2":
        vol = vol.norm(dim=1).mean()
    elif norm_type == "l1":
        vol = vol.sum(dim=1).mean()
    else:
        # vol, _ = torch.diagonal(cov, dim1=-2, dim2=-1).sum(2).max(dim=1)
        raise NotImplementedError
    return vol


class CSQModule(nn.Module):
    def __init__(self, config, num_slots):
        super().__init__()
        concept_dim = config.concept_dim
        self.num_slots = num_slots
        if config.conpcept_projection:
            self.feature2concept = nn.Linear(config.latent_dim, concept_dim)
    
    
    def forward(self,x):
        scores = 1
        node_features = 1
        masks = 1
        outputs = {"scores":scores,"features":node_features,"masks":masks,"match":False}
        return outputs

class CSQNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        concept_dim = config.concept_dim
        construct = ()
        self.base_encoder = AcneKpEncoder(config, indim = 3) # [Encoder]

        gc_dim = config.acne_dim 
        self.decoder = KpDecoder(self.config.acne_num_g, gc_dim,
            self.config.num_pts, self.config)                   # [Decoder]
        self.chamfer_loss = ChamferLoss()
    
        self.csq_modules = [CSQModule(config, num_slots) for num_slots in construct]
        if config.concept_projection:
            self.feature2concept = nn.Linear(config.latent_dim, concept_dim)
        self.scaling = 1.0
    


    def forward(self, inputs):
        pc = inputs['point_cloud'].permute(0,2,1) * self.scaling
        enc_in = inputs['point_cloud'] * self.scaling 

        enc_in = enc_in[...,None].permute(0,2,1,3)

        f_att = self.base_encoder(enc_in, return_att=True)
        gc, attention = f_att
        pose_locals = evaluate_pose(pc , attention)
        kps = pose_locals.squeeze(-1)
        loc_loss = spatial_variance(pc, attention, norm_type="l2")
        # reconstruction from canonical capsules 

        gc = torch.cat([kps[..., None], gc], dim=1)

        y = self.decoder(gc.transpose(2, 1).squeeze(-1))

        chamfer_loss = self.chamfer_loss(pc.permute(0,2,1), y) # Loss


        attention = attention.squeeze(1)
        attention = attention.squeeze(3)

        scene_construct = {"scores":1,"features":1,"masks":1,"raw_features":0,"match":False}

        # [Construct the Hierarchical Representation]
        for csqnet in self.csq_modules:
            csqnet(scene_construct)


        losses = {"chamfer":chamfer_loss,"reconstruction":0.0,"localization":loc_loss}
        outputs = {"loss":losses,}
        return outputs