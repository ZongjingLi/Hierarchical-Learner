import torch
import torch.nn as nn
from .layer_equi import *
import torchvision.models as models

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out

class VNN_DGCNN(nn.Module):
    def __init__(self, c_dim=128, dim=3, hidden_dim=64, k=20):
        super(VNN_DGCNN, self).__init__()
        self.c_dim = c_dim
        self.k = k

        self.conv1 = VNLinearLeakyReLU(2, hidden_dim)
        self.conv2 = VNLinearLeakyReLU(hidden_dim*2, hidden_dim)
        self.conv3 = VNLinearLeakyReLU(hidden_dim*2, hidden_dim)
        self.conv4 = VNLinearLeakyReLU(hidden_dim*2, hidden_dim)

        self.pool1 = meanpool
        self.pool2 = meanpool
        self.pool3 = meanpool
        self.pool4 = meanpool

        self.conv_c = VNLinearLeakyReLU(hidden_dim*4, c_dim, dim=4, share_nonlinearity=True)

    def forward(self, x):

        batch_size = x.size(0)
        x = x.unsqueeze(1).transpose(2, 3)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = self.pool3(x)

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_c(x)
        x = x.mean(dim=-1, keepdim=False)

        return x

class FilterPart(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(7, 2049)
        self.embedding3 = nn.Embedding(10, 2049)
        self.fc = nn.Linear(3000, 1)

    def forward(self, input, idx, out2, query=False, count=False):
        if query:
            idx = torch.tensor([0,1,2,3,4,5,6]).cuda()
        else:
            idx = torch.tensor([idx]).cuda()
        
        output = torch.zeros(idx.size()[0], input.shape[0]).cuda()

        for i in range(idx.size()[0]):
            emb = self.embedding(idx[i]).squeeze()

            emb = emb.unsqueeze(0).expand(input.size())
            output_i = (emb * input)

            output[i] = output_i.sum(-1)

        output = torch.min(output, out2)
        output2 = output.detach().clone()

        if query:
            # output2 = torch.min(output2, out2)
            # output = torch.sigmoid(output)     
            output = torch.max(output, -1)[0]
        # output2 = torch.sigmoid(output2)
        elif count:
            idx = torch.arange(10).cuda()
            acts2 = output_i.unsqueeze(1).expand(-1, 10, -1)
            emb2 = self.embedding3(idx).unsqueeze(0).expand(acts2.size()[0], 10, acts2.size()[-1])
            out3 = ((emb2 * acts2)).sum(-1)
            out3 = torch.softmax(out3, 1)
            # output_ii = output[i].detach().clone()
            out3 = torch.min(out3, out2.unsqueeze(-1).expand(-1, 10)) 
            output = out3
        else:
            # out2 = out2.expand(output2.size())
            # output2 = torch.min(output2, out2)
            output, _ = torch.max(output, 1) 
            # output = torch.softmax(output, 0)
        
        return output, output2

class FilterColor(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(6, 2049)
        self.embedding2 = nn.Embedding(6, 1027)

        self.fc = nn.Linear(3000, 1)

    def forward(self, input, idx, out2, query=False):
        if query:
            idx = torch.tensor([0,1,2,3,4,5]).cuda()
        else:
            idx = torch.tensor([idx]).cuda()

        output = torch.zeros(idx.size()[0], input.shape[0]).cuda()

        for i in range(idx.size()[0]):
            emb = self.embedding2(idx[i]).squeeze()

            emb = emb.unsqueeze(0).expand(input.size())
            output_i = (emb * input)

            output[i] = output_i.sum(-1)

        output2 = output.reshape(-1, output.shape[-1])       
        # output = torch.sigmoid(output)
        if not query:
            output2 = torch.min(output2, out2)
            output, _ = torch.max(output2, 1) 
        else:
            # out2 = out2.expand(output2.size())

            output2 = torch.min(output2, out2)

            output = torch.max(output2, -1)[0]
            # output = torch.softmax(output, 0)

        return output, output2

class CanonicalUnit(nn.Module):
    def __init__(self, config, num_units):
        super().__init__()
        hidden_dim = config.hidden_dim
        kq_dim = 32
        self.config
        self.num_units = num_units
        self.hidden_dim = hidden_dim
        self.kq_dim = kq_dim
    
    def forward(self, x):
        for i in range(10):
            pass
        return x

def mask_localization(points,masks):
    """
    points: [B,N,3]
    masks:  [B,K,N,1]
    """
    N = points.shape[1]
    central_points = torch.einsum("bnu,bknv->bku",points,masks)
    mask_weights = torch.einsum("bknd->bk",masks)
    diffs = (central_points.unsqueeze(2).repeat(1,1,N,1) - points) * masks
    localization_loss = torch.sum(diffs * diffs / mask_weights,dim = 2)
    return {"centers":central_points,"localization_loss":localization_loss}

class HierarchicalVNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        latent_dim = config.latent_dim
        perception = config.perception
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
        if perception == "point_net":
            self.encoder = VNN_ResnetPointnet(c_dim=latent_dim, dim=6) # modified resnet-18
        if perception == "dgcnn":
            self.encoder = VNN_DGCNN(c_dim=latent_dim, dim=6) # modified resnet-18

        self.return_features = True
        self.decoder = DecoderInner(dim=3, z_dim=latent_dim, c_dim=0, hidden_size=latent_dim,leaky=True, \
                        sigmoid=True, return_features=self.return_features, acts="all" )

        self.scaling = config.scaling
        self.obj_map = nn.Linear(1025,100)

    def forward(self, inputs):
        device = self.device
        B, N, _ = inputs["point_cloud"].shape
        enc_in = inputs['point_cloud'] * self.scaling 
        query_points = inputs['coords'] * self.scaling

        enc_in = torch.cat([enc_in, inputs['rgb']], 2)
        z = self.encoder(enc_in)

        outputs = {}

        # [Get Total Query Points and Reconstruction
        all_query_points = torch.cat([query_points, inputs['point_cloud']], dim = 1)
        
        if self.return_features:
            outputs['occ_branch'], outputs['features'], outputs['features2'], outputs['color'], outputs['occ'] = self.decoder(inputs['coords'], inputs['occ'], all_query_points, z)
        else:
            outputs['occ_branch'], outputs['color'], outputs['occ'] = self.decoder(inputs['coords'], inputs['occ'], query_points, z)

        label = inputs['occ'].squeeze(-1)
        label = (label + 1) / 2.

        label = torch.cat([label, torch.ones(B,N)], dim = 1)
        recon_loss_occ= -1 * (label * torch.log(outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - outputs['occ'] + 1e-5)).mean()

        #EPS = 1e-6
        #recon_loss_occ = torch.nn.functional.binary_cross_entropy(torch.clamp(inputs["occ"].to(device) ,min=EPS,max=1-EPS).to(device),outputs["occ"].unsqueeze(-1).to(device))
        query_colors = torch.cat([inputs["coord_color"]], dim = 1)
        recon_loss_color = recon_loss_occ
        #recon_loss_color += torch.nn.functional.mse_loss(outputs["color"].to(device),query_colors.to(device))
        outputs["losses"] = {"occ_reconstruction":recon_loss_occ ,"color_reconstruction":recon_loss_color}
        return outputs

class VNNOccNet(nn.Module):
    def __init__(self,
                 latent_dim,
                 model_type='pointnet',
                 sigmoid=True,
                 return_features=False, 
                 acts='all',
                 scaling=10.0,
                 stage=0):
        super().__init__()

        self.latent_dim = latent_dim
        self.scaling = scaling  # scaling up the point cloud/query points to be larger helps
        self.return_features = return_features

        if model_type == 'dgcnn':
            self.model_type = 'dgcnn'
            self.encoder = VNN_DGCNN(c_dim=latent_dim, dim=6) # modified resnet-18
        else:
            self.model_type = 'pointnet'
            self.encoder = VNN_ResnetPointnet(c_dim=latent_dim, dim=6) # modified resnet-18

        self.decoder = DecoderInner(dim=3, z_dim=latent_dim, c_dim=0, hidden_size=latent_dim, leaky=True, sigmoid=sigmoid, return_features=return_features, acts=acts)

        self.stage = stage

        self.module_1 = FilterPart()
        self.module_2 = FilterColor()

    def forward(self, input):
        out_dict = {}

        enc_in = input['point_cloud'] * self.scaling 
        query_points = input['coords'] * self.scaling 

        enc_in = torch.cat([enc_in, input['rgb']], 2)

        z = self.encoder(enc_in)

        if self.return_features:
            out_dict['occ_branch'], out_dict['features'], out_dict['features2'], out_dict['color'], out_dict['occ'] = self.decoder(input['coords'], input['occ'], query_points, z)
        else:
            out_dict['occ_branch'], out_dict['color'], out_dict['occ'] = self.decoder(input['coords'], input['occ'], query_points, z)

        if not self.stage in [0, 1]:
            bsize = input['question'].size()[0]

            if input['question'][0][1][0] == -1 and input['question'][0][0][0] in [0,1]:
                output2 = out_dict['occ'].detach().clone()
                output3 = torch.ones((bsize, 3000)).cuda()
                out = torch.zeros(bsize).cuda()
                out3 = torch.zeros((bsize, 3000, 10)).cuda()
            else:
                output2 = out_dict['occ'].detach().clone()
                if input['question'][0][-1][0] == 2 or (input['question'][0][0][0] == 2 and input['question'][0][1][0] == -1): 
                    csize = 6
                else:
                    csize = 7
                output2 = output2.unsqueeze(1).expand((bsize, csize, 3000))
                output3 = torch.ones((bsize, csize, 3000)).cuda()

                out = torch.zeros((bsize, csize)).cuda()
                if input['question'][0][1][0] == 4:
                    out3 = torch.zeros((bsize, 3000, 10)).cuda()

            for j in range(bsize):
                for i in range(input['question'][j].shape[0]):
                    
                    if input['question'][j,i,0] == -1:
                        pass
                    else:
                        if input['question'][j,i,0] in [0, 3, 4]:
                            module = self.module_1
                            features = out_dict['features']

                        elif input['question'][j,i,0] in [1, 2]:
                            module = self.module_2
                            features = out_dict['features2']

                        if i == 0:
                            out2 = output2[j].detach().clone()

                        if input['question'][j,i,0] in [2, 3]:
                            out[j], out2 = module(features[j], input['question'][j,i,1], out2, query=True)  

                        elif input['question'][j,i,0] in [4]:
                            o = torch.zeros((3000)).cuda()
                            out2 = torch.where(torch.argmax(out2, 0) == input['question'][j,i,1])[0]

                            o[out2] = 1
                            o = o * out_dict['occ'][j]

                            out3[j], out2 = module(features[j], input['question'][j,i,1], o, query=False, count=True) 
                        else:
                            out[j], out2 = module(features[j], input['question'][j,i,1], out2) 
            
                    if i == 1: 
                        output3[j] = out2
            
            if input['question'][0][1][0] == 4:
                out_dict['ans'] = [out3, o]
            else:
                out_dict['ans'] = [out, output3]

        return out_dict

    def inference(self, input):
        out_dict = {}

        enc_in = input['point_cloud'] * self.scaling 
        query_points = input['coords'] * self.scaling 

        enc_in = torch.cat([enc_in, input['rgb']], 2)

        z = self.encoder(enc_in)

        if self.return_features:
            out_dict['occ_branch'], out_dict['features'], out_dict['features2'], out_dict['color'], out_dict['occ'] = self.decoder(input['coords'], input['occ'], query_points, z)
        else:
            out_dict['occ_branch'], out_dict['color'], out_dict['occ'] = self.decoder(input['coords'], input['occ'], query_points, z)

        bsize = input['question'].size()[0]

        output_2 = out_dict['occ'].detach().clone()
        output_2 = (output_2 > 0.5).float()        

        if input['question'][0][1][0] == 4:
            out3 = torch.zeros((bsize, 3000, 10)).cuda()

        type = ''

        for j in range(bsize):
            for i in range(input['question'][j].shape[0]):

                if input['question'][j,i,0] == -1:
                    pass
                else:
                    if input['question'][j,i,0] == 5:
                        input['question'][j, i, 0] = 1
                        input['question'][j, i, 1] = q   

                    if input['question'][j,i,0] in [0, 3, 4]:
                        module = self.module_1
                        features = out_dict['features']

                    elif input['question'][j,i,0] in [1, 2]:
                        module = self.module_2
                        features = out_dict['features2']

                    if input['question'][j,i,0] in [2, 3]:
                        if input['question'][j,i,0] == 2: 
                            csize = 6
                        else:
                            csize = 7

                        out2 = output_2.squeeze().expand((csize, 3000))

                        out, out2 = module(features[j], input['question'][j,i,1], out2, query=True) 

                        o = torch.argmax(out2, 0)
                        
                        o = o[torch.where(output_2[j])[0]]

                        outs = []
                        for k in range(out.shape[0]):
                            outs.append(torch.sum(o == k))

                        outs = torch.tensor(outs)

                        q = torch.argmax(outs).item()
                        
                        if (i == 1 and len(input['question'][j]) == 2) or (len(input['question'][j]) == 4 and i == 3):
                            correct = (q == input['answer'])
                            type = 'query'
                            # print (out2)
                            # print (torch.argmax(outs), torch.argmax(out), input['answer'], input['question'])

                    elif input['question'][j,i,0] in [4]:
                        o = torch.zeros((3000)).cuda()
                        out2 = torch.where(torch.argmax(out2, 0) == input['question'][j,i,1])[0]

                        o[out2] = 1
                        o = o * (out_dict['occ'][j] > 0.5).float()

                        out3[j], out2 = module(features[j], input['question'][j,i,1], o, query=False, count=True)
                        
                        labels = torch.argmax(out3[j], 1)
                        labels = labels[torch.where(o)[0]]

                        correct = (torch.where(torch.bincount(labels) > 0)[0].shape[0] == input['answer'])
                        type='count'
                        
                    else:
                        if len(input['question'][j]) == 4 and i == 2:
                            output_2[j] = output_3

                        if input['question'][j,i,0] == 0:
                            out2 = output_2[j].unsqueeze(0).expand((7, 3000))
                        else:
                            out2 = output_2[j].unsqueeze(0).expand((6, 3000))

                        _, out2 = module(features[j], input['question'][j,i,1], out2, query=True) 
                        out2 = torch.where(torch.argmax(out2, 0) == input['question'][j,i,1])[0]

                        o = torch.zeros((3000)).cuda()
                        o[out2] = 1

                        if len(input['question'][j]) == 4 and i == 0:
                            output_3 = (1 - o) * output_2[j]

                        o = o * output_2[j]

                        if i == 1:
                            correct = (torch.sum(o) > 0.5 * torch.sum(output_2)) == input['answer']
                            type='filter'
                            break

                        output_2[j] = o  

                        if i in [0, 2] and input['question'][j,i+1,0] == -1:
                            correct = (torch.sum(o) > 100) == input['answer']
                            type='filter'
                            break
                            # print (input['id'], torch.sum(o), input['answer'], input['question'])

        return correct.item(), type                                                                                    

    def extract_latent(self, input):
        enc_in = input['point_cloud'] * self.scaling 
        z = self.encoder(enc_in)
        return z

    def forward_latent(self, z, coords):
        out_dict = {}
        coords = coords * self.scaling 
        out_dict['occ'], out_dict['features'] = self.decoder(coords, z)

        return out_dict['features']

class VNN_ResnetPointnet(nn.Module):
    ''' DGCNN-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output

        self.conv_pos = VNLinearLeakyReLU(3, 128, negative_slope=0.2, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(128, 2*hidden_dim)
        self.block_0 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)

        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.2, share_nonlinearity=False)
        self.pool = meanpool

        if meta_output == 'invariant_latent':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
            self.vn_inv = VNLinear(c_dim, 3)
        elif meta_output == 'equivariant_latent_linear':
            self.vn_inv = VNLinear(c_dim, 3)

    def forward(self, p):
        batch_size = p.size(0)
        p = p.unsqueeze(1).transpose(2, 3)
        #mean = get_graph_mean(p, k=self.k)
        #mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        feat = get_graph_feature_cross(p, k=self.k)

        net = self.conv_pos(feat)

        net = self.pool(net, dim=-1)

        net = self.fc_pos(net)

        net = self.block_0(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))

        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std
        elif self.meta_output == 'equivariant_latent_linear':
            c_std = self.vn_inv(c)
            return c, c_std

        return c

class DecoderInner(nn.Module):
    ''' Decoder class.

    It does not perform any form of normalization.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=128, leaky=False, return_features=False, sigmoid=True, acts='all'):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        self.acts = acts
        if self.acts not in ['all', 'inp', 'first_rn', 'inp_first_rn']:
            #self.acts = 'all'
            raise ValueError('Please provide "acts" equal to one of the following: "all", "inp", "first_rn", "inp_first_rn"')

        # Submodules
        if z_dim > 0:
            self.z_in = VNLinear(z_dim, z_dim)
            # self.z_in = VNLinear(z_dim * 2, z_dim * 2)
        if c_dim > 0:
            self.c_in = VNLinear(c_dim, c_dim)

        self.fc_in = nn.Linear(z_dim*2+c_dim*2+1, hidden_size)
        # self.fc_in = nn.Linear(z_dim*4+c_dim*2+1, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size, hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)
        self.return_features = return_features

        self.fc_out = nn.Linear(hidden_size, 1)
        self.fc_out_branch = nn.Linear(hidden_size, 6)
        # self.fc_out_branch2 = nn.Linear(hidden_size, 3)
        # self.fc_out_branch3 = nn.Linear(hidden_size, 5)

        self.fc_out2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm([hidden_size]),
            nn.LeakyReLU(inplace=True)
        )
        self.fc_out3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm([hidden_size]),
            nn.ReLU(inplace=True)
        )
        self.fc_out4 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm([hidden_size]),
            nn.ReLU(inplace=True)
        )

        # self.attention = AttentionNet()
        # torch.nn.init.kaiming_normal_(self.fc_out2[0].weight)
        # torch.nn.init.kaiming_normal_(self.fc_out3[0].weight)
        # torch.nn.init.kaiming_normal_(self.fc_out4[0].weight)

        self.fc_out5 = nn.Linear(hidden_size, 3)
        self.sigmoid = sigmoid

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.embedding = nn.Embedding(6, 2049)

    def forward(self, coord, occ, p, z, c=None, **kwargs):
        batch_size, T, D = p.size()
        acts = []
        acts2 = []
        acts_inp = []
        acts_first_rn = []
        acts_inp_first_rn = []

        if isinstance(c, tuple):
            c, c_meta = c

        net = (p * p).sum(2, keepdim=True)

        if self.z_dim != 0:
            z = z.view(batch_size, -1, D).contiguous()
            # z = z.view(batch_size, -1, 1).contiguous()
            # net_z = torch.einsum('bmi,bnj->bmn', p, z)
            net_z = torch.einsum('bmi,bni->bmn', p, z)

            z_dir = self.z_in(z)
            z_inv = (z * z_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
            net = torch.cat([net, net_z, z_inv], dim=2)

        if self.c_dim != 0:
            c = c.view(batch_size, -1, D).contiguous()
            net_c = torch.einsum('bmi,bni->bmn', p, c)
            c_dir = self.c_in(c)
            c_inv = (c * c_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
            net = torch.cat([net, net_c, c_inv], dim=2)

        acts.append(net)
        acts_inp.append(net)
        acts_inp_first_rn.append(net)

        net = self.fc_in(net)
        acts.append(net)
        # acts_inp.append(net)
        # acts_inp_first_rn.append(net)

        net = self.block0(net)
        acts.append(net)
        # acts_inp_first_rn.append(net)
        acts_first_rn.append(net)

        net = self.block1(net)
        acts.append(net)
        net = self.block2(net)
        acts.append(net)
        net = self.block3(net)
        acts.append(net)
        net = self.block4(net)
        last_act = net
        acts.append(net)

        out = self.fc_out(self.actvn(net))

        acts2.append(net)
        out2 = self.fc_out2(self.actvn(net))
        acts2.append(out2)
        out2 = self.fc_out3(out2)
        acts2.append(out2)
        out2 = self.fc_out4(out2)
        acts2.append(out2)
        out2 = self.fc_out5(out2)
        acts2.append(out2)
        
        out = out.squeeze(-1)

        acts = torch.cat(acts, dim=-1)
        acts2 = torch.cat(acts2, dim=-1)
        acts = F.normalize(acts, p=2, dim=-1)
        acts2 = F.normalize(acts2, p=2, dim=-1)

        out3 = self.fc_out_branch(self.actvn(net))
        out3 = torch.softmax(out3, 2)

        if self.sigmoid:
            out = F.sigmoid(out)
            out2 = F.sigmoid(out2)

            max_out = out
            
        if self.return_features:
            #acts = torch.cat(acts, dim=-1)
            # if self.acts == 'all':
                
            # elif self.acts == 'inp':
            #     acts = torch.cat(acts_inp, dim=-1)
            # elif self.acts == 'last':
            #     acts = last_act
            # elif self.acts == 'inp_first_rn':
            #     acts = torch.cat(acts_inp_first_rn, dim=-1)
            
            return out3, acts, acts2, out2, max_out
        else:
            return out3, out2, max_out


class DecoderCBatchNorm(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out


class DecoderCBatchNorm2(nn.Module):
    ''' Decoder with CBN class 2.

    It differs from the previous one in that the number of blocks can be
    chosen.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of ResNet blocks
    '''

    def __init__(self, dim=3, z_dim=0, c_dim=128,
                 hidden_size=256, n_blocks=5):
        super().__init__()
        self.z_dim = z_dim
        if z_dim != 0:
            self.fc_z = nn.Linear(z_dim, c_dim)

        self.conv_p = nn.Conv1d(dim, hidden_size, 1)
        self.blocks = nn.ModuleList([
            CResnetBlockConv1d(c_dim, hidden_size) for i in range(n_blocks)
        ])

        self.bn = CBatchNorm1d(c_dim, hidden_size)
        self.conv_out = nn.Conv1d(hidden_size, 1, 1)
        self.actvn = nn.ReLU()

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.conv_p(p)

        if self.z_dim != 0:
            c = c + self.fc_z(z)

        for block in self.blocks:
            net = block(net, c)

        out = self.conv_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out

class DecoderCBatchNormNoResnet(nn.Module):
    ''' Decoder CBN with no ResNet blocks class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.fc_0 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_1 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_2 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_3 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.fc_4 = nn.Conv1d(hidden_size, hidden_size, 1)

        self.bn_0 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_1 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_2 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_3 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_4 = CBatchNorm1d(c_dim, hidden_size)
        self.bn_5 = CBatchNorm1d(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.actvn(self.bn_0(net, c))
        net = self.fc_0(net)
        net = self.actvn(self.bn_1(net, c))
        net = self.fc_1(net)
        net = self.actvn(self.bn_2(net, c))
        net = self.fc_2(net)
        net = self.actvn(self.bn_3(net, c))
        net = self.fc_3(net)
        net = self.actvn(self.bn_4(net, c))
        net = self.fc_4(net)
        net = self.actvn(self.bn_5(net, c))
        out = self.fc_out(net)
        out = out.squeeze(1)

        return out


class DecoderBatchNorm(nn.Module):
    ''' Decoder with batch normalization class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        if self.c_dim != 0:
            self.fc_c = nn.Linear(c_dim, hidden_size)
        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = ResnetBlockConv1d(hidden_size)
        self.block1 = ResnetBlockConv1d(hidden_size)
        self.block2 = ResnetBlockConv1d(hidden_size)
        self.block3 = ResnetBlockConv1d(hidden_size)
        self.block4 = ResnetBlockConv1d(hidden_size)

        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(2)
            net = net + net_c

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(self.bn(net)))
        out = out.squeeze(1)

        return out


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.LeakyReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks 
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(
                c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(
                c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockConv1d(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_h=None, size_out=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = nn.BatchNorm1d(size_in)
        self.bn_1 = nn.BatchNorm1d(size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


# Utility modules
class AffineLayer(nn.Module):
    ''' Affine layer class.

    Args:
        c_dim (tensor): dimension of latent conditioned code c
        dim (int): input dimension
    '''

    def __init__(self, c_dim, dim=3):
        super().__init__()
        self.c_dim = c_dim
        self.dim = dim
        # Submodules
        self.fc_A = nn.Linear(c_dim, dim * dim)
        self.fc_b = nn.Linear(c_dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_A.weight)
        nn.init.zeros_(self.fc_b.weight)
        with torch.no_grad():
            self.fc_A.bias.copy_(torch.eye(3).view(-1))
            self.fc_b.bias.copy_(torch.tensor([0., 0., 2.]))

    def forward(self, x, p):
        assert(x.size(0) == p.size(0))
        assert(p.size(2) == self.dim)
        batch_size = x.size(0)
        A = self.fc_A(x).view(batch_size, 3, 3)
        b = self.fc_b(x).view(batch_size, 1, 3)
        out = p @ A + b
        return out


class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class CBatchNorm1d_legacy(nn.Module):
    ''' Conditional batch normalization legacy layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.fc_gamma = nn.Linear(c_dim, f_dim)
        self.fc_beta = nn.Linear(c_dim, f_dim)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, c):
        batch_size = x.size(0)
        # Affine mapping
        gamma = self.fc_gamma(c)
        beta = self.fc_beta(c)
        gamma = gamma.view(batch_size, self.f_dim, 1)
        beta = beta.view(batch_size, self.f_dim, 1)
        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out

