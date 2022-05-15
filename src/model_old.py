from numpy import histogramdd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from src.hgru_cell import hConvGRUCell
from src.utils import select_scheduler


class PredNet(nn.Module):
    def __init__(self, model_name, n_layers, td_layers, img_layers,
                 pos_layers, ori_layers, bu_channels, td_channels, device) -> None:
        super(PredNet, self).__init__()

        # Model parameters
        self.model_name = model_name
        self.n_layers = n_layers
        self.td_layers = td_layers
        self.img_layers = img_layers
        self.pos_layers = pos_layers
        self.ori_layers = ori_layers
        # if bu_channels[0] != 1:
        #     bu_channels = (1,) + bu_channels[:-1]  # worst coding ever
        self.bu_channels = bu_channels
        self.td_channels = td_channels
        self.do_prediction = not (len(img_layers) == 0)
        self.do_localization = not (len(pos_layers) + len(ori_layers) == 0)
        self.device = device

        # Model states
        self.E_state = [None for _ in range(n_layers)]
        self.R_state = [None for _ in range(n_layers)]

        # Bottom-up connections (bu)
        bu_conv = []
        for l in range(self.n_layers):
            in_channels = 1 if l == 0 else 2 * bu_channels[l - 1]
            bu_conv.append(nn.Conv2d(in_channels, bu_channels[l], kernel_size=3, padding=1))
        self.bu_conv = nn.ModuleList(bu_conv)
        
        # Lateral connections (la)
        la_conv = []
        for l in range(self.n_layers):
            la_conv.append(nn.Conv2d(td_channels[l], bu_channels[l], kernel_size=1, padding=0))
        self.la_conv = nn.ModuleList(la_conv)

        # Top-down connections (td)
        td_upsample, td_conv, td_norm = [], [], []
        for l in range(self.n_layers):
            in_channels = 2 * bu_channels[l] + (0 if l == n_layers - 1 else td_channels[l + 1])
            if l < n_layers - 1:
                td_upsample.append(nn.Upsample(scale_factor=2))
            td_conv.append(hConvGRUCell(in_channels, td_channels[l], kernel_size=5))
            td_norm.append(nn.GroupNorm(td_channels[l], td_channels[l]))
        self.td_upsample = nn.ModuleList(td_upsample)
        self.td_conv = nn.ModuleList(td_conv)
        self.td_norm = nn.ModuleList(td_norm)
    
        # Image prediction
        if self.do_prediction:
            self.img_decoder = Decoder_2D(img_layers, td_channels, 1, nn.Hardtanh(min_val=0.0, max_val=1.0))

        # Target localization
        if self.do_localization:
            self.pos_decoder = Decoder_1D(pos_layers, td_channels, 3, (-0.5, 0.5), nn.Tanh())  # nn.Identity())
            self.ori_decoder = Decoder_1D(ori_layers, td_channels, 6, (-1.0, 1.0), nn.Tanh())  # nn.Identity())

        # Put model on gpu and create folder for the model
        self.to('cuda')
        os.makedirs(f'./ckpt/{model_name}/', exist_ok=True)
        os.makedirs(rf'.\ckpt\{model_name}\pngs', exist_ok=True)
        os.makedirs(rf'.\ckpt\{model_name}\gifs', exist_ok=True)

    def forward(self, A, frame_idx):
        
        # Initialize outputs of this step, as well as internal states, if necessary
        batch_dims = A.size()
        batch_size, _, h, w = batch_dims
        E_pile, R_pile = [None] * self.n_layers, [None] * self.n_layers
        if frame_idx == 0:
            for l in range(self.n_layers):
                self.E_state[l] = torch.zeros(batch_size, 2 * self.bu_channels[l], h, w).cuda()
                self.R_state[l] = torch.zeros(batch_size, self.td_channels[l], h, w).cuda()
                h, w = h // 2, w // 2
        
        # Bottom-up pass
        for l in range(self.n_layers):
            A_hat = F.relu(self.la_conv[l](self.R_state[l]))  # post-synaptic activity of representation prediction
            A = self.bu_conv[l](A)  # presynaptic activity of bottom-up input
            # A = A - A_hat  # pre-synaptic activity, after prediction tries to kill activity in A  # for now in next line
            error = F.relu(torch.cat((A - A_hat, A_hat - A), dim=1))  # post-synaptic activity of A (error goes up)
            E_pile[l] = error  # stored for later: used in top-down pass
            A = F.max_pool2d(error, kernel_size=2, stride=2)  # A update for next bu-layer
            
        # Top-down pass
        for l in reversed(range(self.n_layers)):
            R = self.R_state[l]
            E = self.E_state[l]
            if l < self.n_layers - 1:
                td_input = self.td_upsample[l](self.R_state[l + 1])
                E = torch.cat((E, td_input), dim=1)
            R_pile[l] = self.td_conv[l](E, R)
            R_pile[l] = self.td_norm[l](R_pile[l])

        # Image prediction
        if self.do_prediction:
            img_prediction = self.img_decoder(R_pile)
        else:
            img_prediction = torch.zeros(batch_dims).cuda()

        # Target localization
        if self.do_localization:
            pos = self.pos_decoder(R_pile)
            ori = self.ori_decoder(R_pile)
            tar_localization = torch.cat((pos, ori), dim=1)
        else:
            tar_localization = torch.zeros((batch_size, 9)).cuda()
            
        # Update the states of the network
        self.E_state = E_pile
        self.R_state = R_pile

        # Return the states to the computer
        return E_pile, img_prediction, tar_localization

    def save_model(self, optimizer, scheduler, train_losses, valid_losses, epoch, n_epochs_save):
        ''' Save the model and the training history.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer used to train the model.
        scheduler : torch.optim.lr_scheduler
            Learning rate scheduler used to train the model.
        train_losses : list of float
            Training losses of the model.
        valid_losses : list of float
            Validation losses of the model.

        Returns
        -------
        None
        '''
        ckpt_id = epoch // n_epochs_save * n_epochs_save
        ckpt_path = rf'.\ckpt\{self.model_name}\ckpt_{ckpt_id:03}.pt'
        torch.save({
            'model_name': self.model_name,
            'n_layers': self.n_layers,
            'td_layers': self.td_layers,
            'img_layers': self.img_layers,
            'pos_layers': self.pos_layers,
            'ori_layers': self.ori_layers,
            'bu_channels': self.bu_channels,
            'td_channels': self.td_channels,
            'device': self.device,
            'model_params': self.state_dict(),
            'optimizer_params': optimizer.state_dict(),
            'scheduler_params': scheduler.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses},
            ckpt_path)
        train_losses = [l if l < 10 * sum(train_losses) / len(train_losses) else 0.0 for l in train_losses]
        valid_losses = [l if l < 10 * sum(valid_losses) / len(valid_losses) else 0.0 for l in valid_losses]
        train_axis = list(np.arange(0, epoch + 1, (epoch + 1) / len(train_losses)))[:len(train_losses)]
        valid_axis = list(np.arange(0, epoch + 1, (epoch + 1) / len(valid_losses)))[:len(valid_losses)]
        plt.plot(train_axis, train_losses, label='train')
        plt.plot(valid_axis, valid_losses, label='valid')
        plt.legend()
        plt.savefig(rf'.\ckpt\{self.model_name}\loss_plot.png')
        plt.close()
        print(f'Saved checkpoint at {ckpt_path}')

    @classmethod
    def load_model(cls, model_name, epoch_to_load=None, lr_params=None):
        ''' Load a model from a checkpoint.
        
        Parameters
        ----------
        model_name : str
            Name of the model (used for retrieve the checkpoint folder).
        n_epochs_run : int
            Number of epochs the model has been trained on.
        epoch_to_load : int
            Epoch to load a checkpoint from.
        lr_params : dict
            Learning rate parameters (for optimizer and scheduler).

        Returns
        -------
        model : Model
            Loaded model.
        '''
        ckpt_dir = f'./ckpt/{model_name}/'
        list_dir = [c for c in os.listdir(ckpt_dir) if '.pt' in c]
        ckpt_path = list_dir[-1]  # take last checkpoint (default)
        for ckpt in list_dir:
            if str(epoch_to_load) in ckpt.split('_')[-1]:
                ckpt_path = ckpt
        save = torch.load(ckpt_dir + ckpt_path)
        print(f'Loaded checkpoint at {ckpt_dir + ckpt_path}')
        model = cls(
            model_name=model_name,
            n_layers=save['n_layers'],
            td_layers=save['td_layers'],
            img_layers=save['img_layers'],
            pos_layers=save['pos_layers'],
            ori_layers=save['ori_layers'],
            bu_channels=save['bu_channels'],
            td_channels=save['td_channels'],
            device=save['device'])
        model.load_state_dict(save['model_params'])
        valid_losses = save['valid_losses']
        train_losses = save['train_losses']
        if lr_params is None:
            optimizer, scheduler = None, None
        else:
            optimizer = torch.optim.AdamW(model.parameters(), **lr_params['optimizer'])
            scheduler = select_scheduler(optimizer, lr_params)
            optimizer.load_state_dict(save['optimizer_params'])
            scheduler.load_state_dict(save['scheduler_params'])
        return model, optimizer, scheduler, train_losses, valid_losses


class Decoder_1D(nn.Module):  # decode anything from the latent variables of PredNet
    def __init__(self, decoder_layers, input_channels, n_output_channels, value_range, output_fn):
        super(Decoder_1D, self).__init__()

        self.decoder_layers = decoder_layers
        decoder_shape = 8  # 8
        decoder_features = 64  # 64
        self.pool = nn.AdaptiveAvgPool2d(output_size=(decoder_shape, decoder_shape))
        self.decoded_units = decoder_shape * decoder_shape * decoder_features * len(decoder_layers)
        self.conv = nn.ModuleList([nn.Conv2d(input_channels[l], decoder_features, 1) for l in decoder_layers])
        self.fc = nn.Sequential(nn.Linear(self.decoded_units, n_output_channels), output_fn)
        self.scale = value_range[1] - value_range[0]
        self.bias = value_range[0]

    def forward(self, R_pile):
        hidden_input = torch.cat([self.conv[i](self.pool(R_pile[l])) for i, l in enumerate(self.decoder_layers)], dim=1)
        hidden_input = hidden_input.view((hidden_input.shape[0], self.decoded_units))
        return self.bias + self.scale * self.fc(hidden_input)


class Decoder_2D(nn.Module):  # decode anything from the latent variables of PredNetVGG
    def __init__(self, decoder_layers, input_channels, n_output_channels, output_fn):
        super(Decoder_2D, self).__init__()

        self.decoder_layers = decoder_layers
        decoder_upsp = []
        for l in range(1, max(decoder_layers) + 1):
            inn, out = input_channels[l], input_channels[l - 1]
            decoder_upsp.append(nn.Sequential(
                nn.GroupNorm(inn, inn),
                nn.ConvTranspose2d(in_channels=inn, out_channels=out,
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.GELU()))  # [int(l == max(decoder_layers)):])
        self.decoder_upsp = nn.ModuleList([None] + decoder_upsp)  # None for indexing convenience
        self.decoder_conv = nn.Sequential(
            nn.GroupNorm(input_channels[0], input_channels[0]),
            nn.Conv2d(input_channels[0], n_output_channels, kernel_size=1, bias=False),
            output_fn)
  
    def forward(self, R_pile):

        D = R_pile[max(self.decoder_layers)]  # * self.decoder_prod[-1]
        D = self.decoder_upsp[-1](D) if max(self.decoder_layers) > 0 else self.decoder_conv(D)
        for l in reversed(range(max(self.decoder_layers))):
            if l in self.decoder_layers:
                D = D + R_pile[l]  # * self.decoder_prod[l]
            D = self.decoder_upsp[l](D) if l > 0 else self.decoder_conv(D)
        return D


# Taken from: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# Taken from: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
