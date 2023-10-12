import torch
import torch.nn as nn

from .programs import *
from .percept import *
from .executor import *

from karanir.dklearn import FCBlock, GraphConvolution

class Halearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.perception = CSQNet(config)

        self.executor = SceneProgramExecutor(config)


        self.scene_builder = nn.ModuleList([HierarchyBuilder(config, slot_num) \
            for slot_num in config.hierarchy_construct])
        self.hierarhcy_maps = nn.ModuleList([FCBlock(128,2,config.concept_dim,config.concept_dim)\
            for _ in range(len(config.hierarchy_construct)) ])
            
        self.effective_level = "all"
        self.box_dim = config.concept_dim
        
    
    def forward(self, sample):
        input_pc = sample["point_cloud"]
        
    def build_scene(self,input_features):
        """
        features: BxNxD
        scores:   BxNx1 
        """
        B,N,D = input_features.shape
        scores = []
        features  = []
        connections = []
        box_dim = self.box_dim

        input_scores = torch.ones([B,N,1])
        for i,builder in enumerate(self.scene_builder):
            #input_features [B,N,D]
            if self.is_box:
                masks = builder(input_features[:,:,:box_dim], input_scores, self.executor) # [B,M,N]
            else:masks = builder(input_features, input_scores, self.executor) # [B,M,N]
            # [Build Scene Hierarchy]

            score = torch.max(masks, dim = -1).values.clamp(EPS,1-EPS) # hierarchy scores # [B,M]
            #print(score,masks)
            input_scores = score.unsqueeze(-1)
 
            feature = torch.einsum("bmn,bnd->bmd",masks,input_features) # hierarchy features # [B,M,D]
            if self.is_box:
                feature = self.hierarhcy_maps[i](feature[:,:,:box_dim])
                feature = torch.cat([feature, torch.ones([1,feature.shape[1],box_dim]) * EPS],dim = -1)
            else:
                feature = self.hierarhcy_maps[i](feature)
            # [Build Scores, Features, and Connections]
            scores.append(score) # [B,M]
            features.append(feature) # [B,M,D]
            connections.append(masks) # [B,M,N]

            input_features = feature

        scene_struct = {"scores":scores,"features":features,"connections":connections}
        return scene_struct


class HierarchyBuilder(nn.Module):
    def __init__(self, config, output_slots, nu = 11):
        super().__init__()
        nu = 100 
        num_unary_predicates = nu
        num_binary_predicates = 0
        spatial_feature_dim = 0
        input_dim = num_unary_predicates + spatial_feature_dim + 1
        box = False
        self.num_unary_predicates = num_unary_predicates
        self.num_binary_predicates = num_binary_predicates
        self.graph_conv = GraphConvolution(input_dim,output_slots)
        if box:
            self.edge_predictor = FCBlock(128,3,input_dim * 2,1)
        else: self.edge_predictor = FCBlock(128,3,input_dim *2 ,1)
        self.dropout = nn.Dropout(0.01)
        self.attention_slots = nn.Parameter(torch.randn([1,output_slots,input_dim]))


    def forward(self, x, scores, executor):
        """
        input: 
            x: feature to agglomerate [B,N,D]
        """
        B, N, D = x.shape
        predicates = executor.concept_vocab


        factored_features = torch.cat([x,scores], dim = -1)

        # [Perform Convolution on Factored States]
        GraphFactor = False
        if GraphFactor:
            adjs = self.edge_predictor(
            torch.cat([
                factored_features.unsqueeze(1).repeat(1,N,1,1),
                factored_features.unsqueeze(2).repeat(1,1,N,1),
            ], dim = -1)
            ).squeeze(-1)
            adjs = torch.sigmoid(adjs)
            adjs = self.dropout(adjs)

            adjs = torch.zeros([B, N, N])

            graph_conv_masks = self.graph_conv(factored_features, adjs).permute([0,2,1])
        else:
            #factored_features = F.normalize(factored_features)
            graph_conv_masks = torch.einsum("bnd,bmd->bmn",factored_features,\
                self.attention_slots.repeat(B,1,1)) 
        # [Build Connection Between Input Features and Conv Features]
        M = graph_conv_masks.shape[1]
        scale = 1/math.sqrt(D)
        #scale = 1
        gamma = 0.5

        graph_conv_masks = F.softmax(scale * (graph_conv_masks), dim = -1)

        g = graph_conv_masks
        #g = torch.sigmoid((graph_conv_masks - 0.5) * 10)
        #graph_conv_masks = graph_conv_masks / torch.sum(graph_conv_masks,dim =1,keepdim = True)
        g = g / g.sum( dim = 1, keepdim = True)
        #print(g,scores.squeeze(-1).repeat(1,M,1))

        #print(scores.squeeze(-1).unsqueeze(1).repeat(1,M,1).shape,g.shape)
        g = torch.min(scores.squeeze(-1).unsqueeze(1).repeat(1,M,1),g)

        #g = g /g.sum( dim = 1, keepdim = True)

        return g#graph_conv_masks #[B,10,7]