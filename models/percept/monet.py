import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
from models.nn import *

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, channel_base=64):
        """UNet map in_channel to out_channel with num_block and base

        Args:
            num_blocks (_type_): _description_
            in_channels (_type_): _description_
            out_channels (_type_): _description_
            channel_base (int, optional): channelBase. Defaults to 64.
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.down_convs = nn.ModuleList()
        cur_in_channels = in_channels
        for i in range(num_blocks):
            self.down_convs.append(double_conv(cur_in_channels,
                                               channel_base * 2**i))
            cur_in_channels = channel_base * 2**i

        self.tconvs = nn.ModuleList()
        for i in range(num_blocks-1, 0, -1):
            self.tconvs.append(nn.ConvTranspose2d(channel_base * 2**i,
                                                  channel_base * 2**(i-1),
                                                  2, stride=2))

        self.up_convs = nn.ModuleList()
        for i in range(num_blocks-2, -1, -1):
            self.up_convs.append(double_conv(channel_base * 2**(i+1), channel_base * 2**i))

        self.final_conv = nn.Conv2d(channel_base, out_channels, 1)

    def forward(self, x):
        intermediates = []
        cur = x
        for down_conv in self.down_convs[:-1]:
            cur = down_conv(cur)
            intermediates.append(cur)
            cur = nn.MaxPool2d(2)(cur)

        cur = self.down_convs[-1](cur)

        for i in range(self.num_blocks-1):
            cur = self.tconvs[i](cur)
            cur = torch.cat((cur, intermediates[-i -1]), 1)
            cur = self.up_convs[i](cur)

        return self.final_conv(cur)


class AttentionNet(nn.Module):
    def __init__(self, blocks,base):
        super().__init__()
        self.unet = UNet(num_blocks= blocks,
                         in_channels=3 + 1,
                         out_channels=2,
                         channel_base= base)

    def forward(self, x, scope):
        inp = torch.cat((x, scope), 1)
        logits = self.unet(inp)
        alpha = torch.softmax(logits, 1)
        # output channel 0 represents alpha_k,
        # channel 1 represents (1 - alpha_k).
        mask = scope * alpha[:, 0:1]
        new_scope = scope * alpha[:, 1:2]
        return mask, new_scope

class EncoderNet(nn.Module):
    def __init__(self, width, height,inchannle = 3,latent_dim = 128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inchannle, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True)
        )

        for i in range(4):
            width = (width - 1) // 2
            height = (height - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(64 * width * height, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim)
        )
        self.mlp = FCBlock(200,3,64 * width * height,latent_dim)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)

        x = self.mlp(x)
        return x

class DecoderNet(nn.Module):
    def __init__(self, width, height,in_channels,out_channels = 3):
        super().__init__()
        self.height = height
        self.width = width
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels + 2, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )
        ys = torch.linspace(-1, 1, self.height + 8)
        xs = torch.linspace(-1, 1, self.width + 8)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)
        #self.T = FCBlock(128,3,in_channels,in_channels)

    def forward(self, z):
        #z = self.T(z)
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8, self.width + 8)
        coord_map = 10*self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)

        result = self.convs(inp)
        return result


class Monet(nn.Module):
    def __init__(self, height, width,channel,base = 64):
        super().__init__()
        self.channel = channel
        self.attention = AttentionNet(3,base)
        self.encoder = EncoderNet(height, width,self.channel + 1,base * 2)
        self.decoder = DecoderNet(height, width,base,self.channel)
        self.beta = 0.5
        self.gamma = 0.25
        self.base = base
        self.num = 4
        

    def forward(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(-1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        masks.append(scope)
        
        loss = torch.zeros_like(x[:, 0, 0, 0])
        mask_preds = []
        full_reconstruction = torch.zeros_like(x)
        
        p_xs = torch.zeros_like(loss)
        kl_zs = torch.zeros_like(loss)
        for i, mask in enumerate(masks):
            z, kl_z = self.__encoder_step(x, mask)
            sigma = 0.2 if i == 0 else 0.8
            p_x, x_recon, mask_pred = self.__decoder_step(x, z, mask, sigma)
            mask_preds.append(mask_pred)
            loss += -p_x + self.beta * kl_z
            p_xs += -p_x
            kl_zs += kl_z
            full_reconstruction += mask * x_recon

        masks = torch.cat(masks, 1)
        tr_masks = torch.transpose(masks, 1, 3)
        q_masks = dists.Categorical(probs=tr_masks)
        q_masks_recon = dists.Categorical(logits=torch.stack(mask_preds, 3))
        kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        kl_masks = torch.sum(kl_masks, [1, 2])
        # print('px', p_xs.mean().item(),
        #       'kl_z', kl_zs.mean().item(),
        #       'kl masks', kl_masks.mean().item())
        loss += self.gamma * kl_masks
        return {'loss': loss,
                'masks': masks,
                'reconstructions': full_reconstruction}


    def __encoder_step(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        q_params = self.encoder(encoder_input)
        means = torch.sigmoid(q_params[:, :self.base]) * 6 - 3 
        sigmas = torch.sigmoid(q_params[:, self.base:]) * 3
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + dist_0.sample()
        q_z = dist.log_prob(z)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)
        return z, kl_z

    def __decoder_step(self, x, z, mask, sigma):
        decoder_output = self.decoder(z)
        x_recon = torch.sigmoid(decoder_output[:, :self.channel])
        mask_pred = decoder_output[:, 3]
        dist = dists.Normal(x_recon, sigma)
        p_x = dist.log_prob(x)
        p_x *= mask
        p_x = torch.sum(p_x, [1, 2, 3])
        return p_x, x_recon, mask_pred


class ObjectPriorNet(nn.Module):
    def __init__(self,width,height,base = 128):
        super().__init__()
        self.encoder_net = EncoderNet(width,height,4,base)
        self.feature_net = FCBlock(200,4,base,base)
        self.score_net = FCBlock(200,4,base,1)
    def forward(self,image,masks):
        masked_image = image * masks

        features = self.encoder_net(masked_image)
        features = self.feature_net(features)
        return features,self.score_net(features)

class MonetPerception(nn.Module):
    def __init__(self, config):
        super().__init__()
        channel = 4
        width,height = config.resolution
        base = config.hidden_dim
        self.object_prior =ObjectPriorNet(width,height,base)
        self.channel = channel
        self.attention = AttentionNet(3,base)
        self.encoder = EncoderNet(height, width,self.channel + 1,base * 2)
        self.decoder = DecoderNet(height, width,base,self.channel)
        self.score_prior = FCBlock(128,3,base,1)
        self.feature_prior = FCBlock(128,3,base,config.object_dim)
        self.beta = 0.5
        self.gamma = 0.25
        self.base = base
        self.num = config.slots
        

    def forward(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(self.num):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        masks.append(scope)
        
        loss = torch.zeros_like(x[:, 0, 0, 0])
        mask_preds = []
        full_reconstruction = torch.zeros_like(x)
        
        masks = torch.cat(masks, 1)
        features,scores = self.object_prior(x,masks.unsqueeze(2)[0]) 
        raw_scores = torch.sigmoid(scores * 5).reshape([1,-1])
        scores = torch.cat([1-raw_scores[0:1],raw_scores[1:]],0)
  
        p_xs = torch.zeros_like(loss)
        kl_zs = torch.zeros_like(loss)
        for i in range(masks.shape[1]):
            mask = masks[0][i:i+1].unsqueeze(0)
            z, kl_z = self.__encoder_step(x, mask)
            #osg = 0.12
            osg = 0.8
            sigma = 0.06 if i == 0 else osg
            #print(x.shape,z.shape,mask.shape)
            p_x, x_recon, mask_pred = self.__decoder_step(x, z, mask, sigma)
            mask_preds.append(mask_pred)
            loss += -p_x + self.beta * kl_z
            p_xs += -p_x
            kl_zs += kl_z
            if (i == 0):
                full_reconstruction += mask * x_recon
            else:
                full_reconstruction += mask * x_recon * scores[0][i]

        tr_masks = torch.transpose(masks, 1, 3)
        q_masks = dists.Categorical(probs=tr_masks)
        q_masks_recon = dists.Categorical(logits=torch.stack(mask_preds, 3))
        kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        kl_masks = torch.sum(kl_masks, [1, 2])
        # print('px', p_xs.mean().item(),
        #       'kl_z', kl_zs.mean().item(),
        #       'kl masks', kl_masks.mean().item())
        loss += self.gamma * kl_masks
        masks = masks.unsqueeze(2)
        #print(features[1:].shape,scores[:,1:].shape,masks[:,1:].shape)
        features = self.feature_prior(features)
        return {"loss":loss,"full_recons":full_reconstruction,
        "object_features":features[1:,:],
        "scores":scores.reshape([-1,1])[1:,:],
        "masks":masks[:,1:,:,:,:],
        "recons":None}
        return loss,features[1:],scores[:,1:],{'loss': loss,
                'masks': masks[:,1:],
                'reconstructions': full_reconstruction}


    def __encoder_step(self, x, mask):

        encoder_input = torch.cat((x, mask), 1)

        q_params = self.encoder(encoder_input)
        means = torch.sigmoid(q_params[:, :self.base]) * 6 - 3 
        sigmas = torch.sigmoid(q_params[:, self.base:]) * 3
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + dist_0.sample()
        q_z = dist.log_prob(z)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)
        return z, kl_z

    def __decoder_step(self, x, z, mask, sigma):
        decoder_output = self.decoder(z)
        x_recon = torch.sigmoid(decoder_output[:, :self.channel])
        mask_pred = decoder_output[:, 3]
        dist = dists.Normal(x_recon, sigma)
        p_x = dist.log_prob(x)
        p_x *= mask
        p_x = torch.sum(p_x, [1, 2, 3])
        return p_x, x_recon, mask_pred