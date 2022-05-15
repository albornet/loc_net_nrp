import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from src.hgru_cell import hConvGRUCell
from src.utils import select_scheduler


class PredNet(nn.Module):
    def __init__(self, model_name, n_layers, td_layers, img_layers,
                 pos_layers, ori_layers, bu_channels, td_channels, device) -> None:
        ''' Create a PredNet model, initialize its states, create a checkpoint
            folder and put the model on the correct device (cpu or gpu).
        
        Parameters
        ----------
        model_name : str
            Name of the model (to identify the checkpoint folder when loading).
        n_layers : int
            Number of layers in the bottom-up and top-down networks.
        td_layers : list of str
            Type of cell used in the top-down computations.
        pos_layers : list of int
            What td_layers are used by the position decoder.
        ori_layers : list of int
            What td_layers are used by the orientationdecoder.
        bu_channels : list of int
            Number of channels in the bottom-up layers.
        td_channels : list of int
            Number of channels in the top-down layers.
        do_localization : bool
            Whether to decode localization.
        device : torch.device
            Device to use for the computation ('cpu', 'cuda').

        Returns
        -------
        None.
        '''
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
        for l in range(self.n_layers - 1):  # "2", because error is torch.cat([pos, neg])
            bu_conv.append(nn.Conv2d(2 * bu_channels[l], bu_channels[l + 1], kernel_size=5, padding=2))
        self.bu_conv = nn.ModuleList(bu_conv)
        
        # Lateral connections (la)
        la_conv = []
        for l in range(self.n_layers):
            la_conv.append(nn.Conv2d(td_channels[l], bu_channels[l], kernel_size=1, padding=0))
        self.la_conv = nn.ModuleList(la_conv)

        # Top-down connections (td)
        td_conv, td_norm = [], []
        for l in range(self.n_layers):  # "2", because error is torch.cat([pos, neg])
            in_channels = 2 * bu_channels[l] + (td_channels[l + 1] if l < n_layers - 1 else 0)
            if td_layers[l] == 'H':
                td_conv.append(hConvGRUCell(in_channels, td_channels[l], kernel_size=5))  # implicit padding
            elif td_layers[l] == 'C':
                td_conv.append(nn.Conv2d(in_channels, td_channels[l], kernel_size=5, padding=2))
            td_norm.append(nn.GroupNorm(td_channels[l], td_channels[l]))
        self.td_conv = nn.ModuleList(td_conv)
        self.td_norm = nn.ModuleList(td_norm)
        self.td_upsample = nn.ModuleList([nn.Upsample(scale_factor=2) for _ in range(n_layers - 1)])

        # Image prediction
        if self.do_prediction:
            self.img_decoder = Decoder_2D(img_layers, td_channels, 1, nn.Hardtanh(min_val=0.0, max_val=1.0))

        # Target localization
        if self.do_localization:
            self.pos_decoder = Decoder_1D(pos_layers, td_channels, 64, 3, (-0.5, 0.5), nn.Sigmoid())
            self.ori_decoder = Decoder_1D(ori_layers, td_channels, 64, 6, (-1.0, 1.0), nn.Sigmoid())

        # Put model on gpu and create folder for the model
        self.to(device)
        os.makedirs(rf'.\ckpt\{model_name}', exist_ok=True)
        os.makedirs(rf'.\ckpt\{model_name}\pngs', exist_ok=True)
        os.makedirs(rf'.\ckpt\{model_name}\gifs', exist_ok=True)

    def forward(self, A, frame_idx):
        ''' Forward pass of the PredNet.
        
        Parameters
        ----------
        A : torch.Tensor
            Input image (from a batch of input sequences).
        frame_idx : int
            Index of the current frame in the sequence.

        Returns
        -------
        E_pile : list of torch.Tensor
            Activity of all errors units (bottom-up pass).
        img_prediction : torch.Tensor
            Prediction of the next frame input (first layer of the network).
        pos_prediction : torch.Tensor
            Prediction of the position of the target (if do_localization is True).
        ori_prediction : torch.Tensor
            Prediction of the orientation of the target (if do_localization is True).
        '''
        # Initialize outputs of this step, as well as internal states, if necessary
        batch_dims = A.size()
        batch_size, _, h, w = batch_dims
        E_pile, R_pile = [None] * self.n_layers, [None] * self.n_layers
        if frame_idx == 0:
            for l in range(self.n_layers):
                self.E_state[l] = torch.zeros(batch_size, 2 * self.bu_channels[l], h, w).to(self.device)
                self.R_state[l] = torch.zeros(batch_size, self.td_channels[l], h, w).to(self.device)
                h, w = h // 2, w // 2

        # Bottom-up pass
        for l in range(self.n_layers):
            A_hat = F.relu(self.la_conv[l](self.R_state[l]))  # post-synaptic activity of representation prediction
            error = F.relu(torch.cat((A - A_hat, A_hat - A), dim=1))  # post-synaptic activity of A (error goes up)
            E_pile[l] = error  # stored for next step, used in top-down pass
            A = F.max_pool2d(error, kernel_size=2, stride=2)  # A update for next bu-layer
            if l < self.n_layers - 1:
                A = self.bu_conv[l](A)  # presynaptic activity of bottom-up input
            # if l == 0:
            #     img_prediction = F.hardtanh(A_hat, min_val=0.0, max_val=1.0)

        # Top-down pass
        for l in reversed(range(self.n_layers)):
            td_input = self.E_state[l]
            if l < self.n_layers - 1:
                td_output = self.td_upsample[l](self.R_state[l + 1])
                td_input = torch.cat((td_input, td_output), dim=1)
            if self.td_layers[l] == 'H':
                R_pile[l] = self.td_norm[l](self.td_conv[l](td_input, self.R_state[l]))
            elif self.td_layers[l] == 'C':
                R_pile[l] = self.td_norm[l](self.td_conv[l](td_input))

        # Image prediction
        if self.do_prediction:
            img_prediction = self.img_decoder(R_pile)
        else:
            img_prediction = torch.zeros(batch_dims).cuda()
        
        # Target localization (3-dof position, 6-dof orientation)
        if self.do_localization:
            pos = self.pos_decoder(R_pile)
            ori = self.ori_decoder(R_pile)
            tar_localization = torch.cat((pos, ori), dim=1)
        else:
            tar_localization = torch.zeros((batch_size, 3 + 6)).cuda()
            
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
            rf'.\ckpt\{self.model_name}\ckpt_{ckpt_id:03}.pt')
        train_losses = [l if l < 10 * sum(train_losses) / len(train_losses) else 0.0 for l in train_losses]
        valid_losses = [l if l < 10 * sum(valid_losses) / len(valid_losses) else 0.0 for l in valid_losses]
        train_axis = list(np.arange(0, epoch + 1, (epoch + 1) / len(train_losses)))[:len(train_losses)]
        valid_axis = list(np.arange(0, epoch + 1, (epoch + 1) / len(valid_losses)))[:len(valid_losses)]
        plt.plot(train_axis, train_losses, label='train')
        plt.plot(valid_axis, valid_losses, label='valid')
        plt.legend()
        plt.savefig(rf'.\ckpt\{self.model_name}\loss_plot.png')
        plt.close()

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


class Decoder_1D(nn.Module):
    def __init__(self, decoder_layers, input_channels, n_hidden_channels,
                 n_output_channels, value_range, output_fn):
        ''' Decode any 1D quantity from the latent variables of PredNet
        
        Parameters
        ----------
        decoder_layers : list of int
            Layer indices of the PredNet that are sent to the decoder.
        input_channels : int
            Number of channels for each decoded layers.
        n_output_channels : int
            Number of decoded outputs.
        value_range : tuple
            Range for the output.
        '''
        super(Decoder_1D, self).__init__()
        self.value_range = value_range
        self.decoder_layers = decoder_layers
        decoded_units = sum([(32 // (2 ** l)) ** 2 * input_channels[l] for l in decoder_layers])
        self.fc = nn.Sequential(nn.Linear(decoded_units, n_output_channels), output_fn)
        
    def forward(self, R_pile):
        ''' Forward pass of the decoder.
        
        Parameters
        ----------
        R_pile : torch.Tensor
            Tensor of shape (batch_size, n_layers, n_channels, n_channels, n_channels)
            containing the latent variables of the PredNet.
        
        Returns
        -------
        decoded_output : torch.Tensor
            Tensor of shape (batch_size, n_output_channels) containing the decoded output.
        '''
        decoded_output = [torch.flatten(R_pile[l], start_dim=1) for l in self.decoder_layers]
        decoded_output = self.fc(torch.cat(decoded_output, dim=1))
        if self.value_range is not None:
            decoded_output = decoded_output * (self.value_range[1] - self.value_range[0])
            decoded_output = decoded_output + self.value_range[0]
        return decoded_output


class Decoder_2D(nn.Module):
    def __init__(self, decoder_layers, input_channels, n_output_channels, output_fn):
        super(Decoder_2D, self).__init__()
        self.decoder_layers = decoder_layers
        decoder_upsp = []
        for l in range(1, max(decoder_layers) + 1):
            inn, out = input_channels[l], input_channels[l - 1]
            decoder_upsp.append(nn.Sequential(
                nn.GroupNorm(inn, inn),
                nn.ConvTranspose2d(in_channels=inn,
                                   out_channels=out,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias=False),
                nn.GELU()))  # [int(l == max(decoder_layers)):])
        self.decoder_upsp = nn.ModuleList([None] + decoder_upsp)  # None for indexing convenience
        self.decoder_conv = nn.Sequential(nn.GroupNorm(input_channels[0],
                                                       input_channels[0]),
                                          nn.Conv2d(input_channels[0],
                                                    n_output_channels,
                                                    kernel_size=1,
                                                    bias=False),
                                          output_fn)
  
    def forward(self, R_pile):
        D = R_pile[max(self.decoder_layers)]  # * self.decoder_prod[-1]
        D = self.decoder_upsp[-1](D) if max(self.decoder_layers) > 0 else self.decoder_conv(D)
        for l in reversed(range(max(self.decoder_layers))):
            if l in self.decoder_layers:
                D = D + R_pile[l]  # * self.decoder_prod[l]
            D = self.decoder_upsp[l](D) if l > 0 else self.decoder_conv(D)
        return D
