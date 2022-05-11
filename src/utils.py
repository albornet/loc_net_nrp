import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import numpy as np
from src.loss_fn import FocalLoss, DiceLoss
bce_loss_fn = nn.BCEWithLogitsLoss()
mse_loss_fn = nn.MSELoss()
mae_loss_fn = nn.L1Loss()
foc_loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
dice_loss_fn = DiceLoss()

def train_fn(train_dl, model, optimizer, loss_weight, t_start,
             n_backprop_frames, epoch, plot_gif=True):

    # Train the network for one epoch
    model.train()
    plot_loss_train = []
    n_batches = len(train_dl)
    with torch.autograd.set_detect_anomaly(True):

        for batch_idx, (images, L_lbls) in enumerate(train_dl):
            batch_loss_train = 0.0
            A_seq, P_seq, L_seq, L_lbl_seq = [], [], [], []
            n_frames = images.shape[-1]
            loss = 0.0

            for t in range(n_frames):
                A = images[..., t].to(device=model.device)
                L_lbl = L_lbls[..., t].to(device='cuda')
                E, P, L = model(A, t)
                A_seq.append(A.detach().cpu())
                P_seq.append(P.detach().cpu())
                L_seq.append(L.detach().cpu())
                L_lbl_seq.append(L_lbl.detach().cpu())
                time_weight = float(t >= t_start)
                loss = loss + loss_fn(E, A, P, L, L_lbl,
                                      time_weight, loss_weight, batch_idx, n_batches)

                if (t + 1) % n_backprop_frames == 0:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # slowdown?
                    optimizer.step()
                    model.E_state = [s.detach() if s is not None else s for s in model.E_state]
                    model.R_state = [s.detach() if s is not None else s for s in model.R_state]
                    batch_loss_train += loss.detach().item() / n_frames
                    loss = 0.0
                
            plot_loss_train.append(batch_loss_train)  # += batch_loss_train / n_batches
            if ((epoch == 0 and (batch_idx % 10) == 0) or (batch_idx == 0)) and plot_gif:
                A_seq = torch.stack(A_seq, axis=-1)
                P_seq = torch.stack(P_seq, axis=-1)
                L_seq = torch.stack(L_seq, axis=-1)
                L_lbl_seq = torch.stack(L_lbl_seq, axis=-1)
                plot_recons(A_seq, L_lbl_seq, P_seq, L_seq, epoch=epoch,
                    output_dir=f'./ckpt/{model.model_name}/')
                
    print(f'\r\nEpoch train loss : {sum(plot_loss_train) / len(plot_loss_train)}')
    return plot_loss_train


def valid_fn(valid_dl, model, loss_weight, t_start, epoch, plot_gif=True):

    model.eval()
    plot_loss_valid = []  # 0.0
    n_batches = len(valid_dl)
    with torch.no_grad():
        for batch_idx, (images, L_lbls) in enumerate(valid_dl):
            batch_loss_valid = 0.0
            A_seq, P_seq, L_seq, L_lbl_seq = [], [], [], []
            n_frames = images.shape[-1]
            for t in range(n_frames):
                A = images[..., t].to(device=model.device)
                L_lbl = L_lbls[..., t].to(device=model.device)
                E, P, L = model(A, t)                
                A_seq.append(A.detach().cpu())
                P_seq.append(P.detach().cpu())
                L_seq.append(L.detach().cpu())
                L_lbl_seq.append(L_lbl.detach().cpu())
                time_weight = float(t >= t_start)
                loss = loss_fn(E, A, P, L, L_lbl, time_weight,
                               loss_weight, batch_idx, n_batches)
                batch_loss_valid += loss.item() / n_frames
            plot_loss_valid.append(batch_loss_valid)  # += batch_loss_valid / n_batches
            if ((epoch == 0 and (batch_idx % 10) == 0) or (batch_idx == 0)) and plot_gif:
                A_seq = torch.stack(A_seq, axis=-1)
                P_seq = torch.stack(P_seq, axis=-1)
                L_seq = torch.stack(L_seq, axis=-1)
                L_lbl_seq = torch.stack(L_lbl_seq, axis=-1)
                plot_recons(
                    A_seq, L_lbl_seq, P_seq, L_seq, epoch=epoch,
                    output_dir=f'./ckpt/{model.model_name}/',
                    mode='test' if epoch == -1 else 'valid')

    print(f'\r\nEpoch valid loss: {sum(plot_loss_valid) / len(plot_loss_valid)}\n')
    return plot_loss_valid


def loss_fn(E, frame, P, L, L_lbl, time_weight, loss_weight, batch_idx, n_batches):

    # Latent prediction error loss (unsupervised)
    latent_loss = 0.0 if E is None else sum([torch.mean(E[l]) * w for l, w in enumerate(loss_weight['latent'])])
    
    # Image prediction loss (unsupervised)
    img_mae_loss = mae_loss_fn(P, frame) * loss_weight['prd_mae'] if loss_weight['prd_mae'] else 0.0
    img_mse_loss = mse_loss_fn(P, frame) * loss_weight['prd_mse'] if loss_weight['prd_mse'] else 0.0
    img_bce_loss = bce_loss_fn(P, frame) * loss_weight['prd_bce'] if loss_weight['prd_bce'] else 0.0

    # Localization prediction loss (supervised)
    loc_mae_loss = mae_loss_fn(L, L_lbl) * loss_weight['loc_mae'] if loss_weight['loc_mae'] else 0.0
    loc_mse_loss = mse_loss_fn(L, L_lbl) * loss_weight['loc_mse'] if loss_weight['loc_mse'] else 0.0
    loc_bce_loss = bce_loss_fn(L, L_lbl) * loss_weight['loc_bce'] if loss_weight['loc_bce'] else 0.0
   
    # Total loss
    img_loss = img_mae_loss + img_mse_loss + img_bce_loss
    loc_loss = loc_mae_loss + loc_mse_loss + loc_bce_loss
    total_loss = latent_loss + img_loss + (loc_loss if loc_loss > 0 else 0.0)
    print(
        f'\rBatch ({batch_idx + 1}/{n_batches}) - loss: {total_loss:.3f} [' +
        f'latent: {latent_loss:.3f}, ' +
        f'image: {img_loss:.3f} (mae: {img_mae_loss:.3f}, mse: {img_mse_loss:.3f}, bce: {img_bce_loss:.3f}) ' +
        f'loc: {loc_loss:.3f} (mae: {loc_mae_loss:.3f}, mse: {loc_mse_loss:.3f}, bce: {loc_bce_loss:.3f})]', end='')
    return total_loss * time_weight


def plot_recons(A_seq, L_lbl_seq, P_seq, L_seq,
    epoch=0, sample_indexes=(0,), output_dir='./', mode='train'):

    batch_size, n_channels, n_rows, n_cols, n_frames = A_seq.shape
    img_plot = A_seq.numpy()
    pred_plot = P_seq.numpy()

    L_seq = L_seq.view((batch_size, n_channels, L_seq.shape[-2] // 3, 3, n_frames))
    L_lbl_seq = L_lbl_seq.view((batch_size, n_channels, L_lbl_seq.shape[-2] // 3, 3, n_frames))   
    L_min, L_max = -1.5, 3.0  # to check with the dataset
    L_seq = (L_seq - L_min) / (L_max - L_min)
    L_lbl_seq = (L_lbl_seq - L_min) / (L_max - L_min)
    L_seq = [F.interpolate(L_seq[..., t], size=img_plot.shape[2:4]) for t in range(n_frames)]
    L_lbl_seq = [F.interpolate(L_lbl_seq[..., t], size=img_plot.shape[2:4]) for t in range(n_frames)]
    loc_plot = torch.stack(L_seq, dim=-1).numpy()
    loc_lbl = torch.stack(L_lbl_seq, dim=-1).numpy()

    rect_width = 1
    h_rect = np.ones((batch_size, n_channels, rect_width, n_cols, n_frames))
    v_rect = np.ones((batch_size, n_channels, 2 * n_rows + rect_width, rect_width, n_frames))
    img_data = np.concatenate((img_plot, h_rect, pred_plot), axis=2)
    loc_data = np.concatenate((loc_lbl, h_rect, loc_plot), axis=2)
    out_batch = np.concatenate((img_data, v_rect, loc_data), axis=3)
    out_batch = out_batch.transpose((0, 2, 3, 1, 4))[:, :, :, 0, :]  # remove "color" channel

    for s_idx in sample_indexes:
        out_seq = out_batch[s_idx]
        gif_frames = [(255. * out_seq[..., t]).astype(np.uint8) for t in range(n_frames)]
        gif_path = f'{output_dir}{mode}_epoch{epoch:02}_id{s_idx:02}'
        imageio.mimsave(f'{gif_path}.gif', gif_frames, duration=0.1)
