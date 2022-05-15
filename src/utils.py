import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import numpy as np
import matplotlib.pyplot as plt
from src.dataset_fn_multi import six_dof_to_euler
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

bce_loss_fn = nn.BCEWithLogitsLoss()
mse_loss_fn = nn.MSELoss()
mae_loss_fn = nn.L1Loss()

def train_fn(train_dl, model, optimizer, scheduler, loss_params, epoch, plot_gif=True):

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
                L_lbl = L_lbls[..., t].to(device=model.device)
                E, P, L = model(A, t)
                A_seq.append(A.detach().cpu())
                P_seq.append(P.detach().cpu())
                L_seq.append(L.detach().cpu())
                L_lbl_seq.append(L_lbl.detach().cpu())
                time_weight = float(t >= loss_params['t_start'])  # 0.0 if t < loss_params['t_start'] else (1.0 / t)
                loss = loss + loss_fn(E, A, P, L, L_lbl,
                                      time_weight, loss_params, batch_idx, n_batches)

                if (t + 1) % loss_params['n_backprop_frames'] == 0:
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

            if type(scheduler) is torch.optim.lr_scheduler.OneCycleLR:
                scheduler.step()
            elif batch_idx == (len(train_dl) - 1):
                scheduler.step()
                
    print(f'\r\nEpoch train loss : {sum(plot_loss_train) / len(plot_loss_train)}')
    return plot_loss_train


def valid_fn(valid_dl, model, loss_params, epoch, plot_gif=True):

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
                time_weight = float(t >= loss_params['t_start'])  # 0.0 if t < loss_params['t_start'] else (1.0 / t)
                loss = loss_fn(E, A, P, L, L_lbl, time_weight,
                               loss_params, batch_idx, n_batches)
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


def loss_fn(E, frame, P, L, L_lbl, time_weight, loss_params, batch_idx, n_batches):

    # Latent prediction error loss (unsupervised)
    latent_loss = 0.0 if E is None else sum([torch.mean(E[l]) * w for l, w in enumerate(loss_params['latent'])])
    
    # Image prediction loss (unsupervised)
    img_mae_loss = mae_loss_fn(P, frame) * loss_params['prd_mae'] if loss_params['prd_mae'] else 0.0

    # Localization prediction loss (supervised)
    loc_mae_loss = mae_loss_fn(L, L_lbl) * loss_params['loc_mae'] if loss_params['loc_mae'] else 0.0
    loc_mse_loss = mse_loss_fn(L, L_lbl) * loss_params['loc_mse'] if loss_params['loc_mse'] else 0.0
    loc_bce_loss = bce_loss_fn(L, L_lbl) * loss_params['loc_bce'] if loss_params['loc_bce'] else 0.0
   
    # Total loss
    img_loss = img_mae_loss
    loc_loss = loc_mae_loss + loc_mse_loss + loc_bce_loss
    total_loss = latent_loss + img_loss + (loc_loss if loc_loss > 0 else 0.0)
    print(
        f'\rBatch ({batch_idx + 1}/{n_batches}) - loss: {total_loss:.3f} [' +
        f'latent: {latent_loss:.3f}, image: {img_loss:.3f} (mae: {img_mae_loss:.3f}) ' +
        f'loc: {loc_loss:.3f} (mae: {loc_mae_loss:.3f}, mse: {loc_mse_loss:.3f}, bce: {loc_bce_loss:.3f})]', end='')
    return total_loss * time_weight


def plot_recons(A_seq, L_lbl_seq, P_seq, L_seq, epoch=0, sample_indexes=(0,), output_dir='./', mode='train'):

    fig, ax = plot_loc_sequence(L_seq, L_lbl_seq)
    for s_idx in sample_indexes:
        png_path = rf'{output_dir}\pngs\{mode}_epoch{epoch:02}_id{s_idx:02}.png'
        fig.savefig(png_path)
    plt.close()

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
        gif_path = rf'{output_dir}\gifs\{mode}_epoch{epoch:02}_id{s_idx:02}.gif'
        imageio.mimsave(gif_path, gif_frames, duration=0.1)


def plot_loc_sequence(pred_sequence, true_sequence, return_fig=True):
    ''' Plots 3D-localization predictions.
    
    Parameters
    ----------
    pred_sequence : torch.tensor of shape (batch_size, 3 + 6, n_frames)
        containing the localization predictions (3-dof position and 6-dof orientation)
    true_sequence : torch.tensor of shape (batch_size, 3 + 6, n_frames)
        containing the true localizations (3-dof position and 6-dof orientation)
    
    Returns
    -------
    None
    '''
    pos_pred_sequence = pred_sequence[0, :3, :].transpose(0, 1)
    ori_pred_sequence = pred_sequence[0, 3:, :].transpose(0, 1)
    ori_pred_sequence = six_dof_to_euler(ori_pred_sequence)
    pos_pred_sequence_array = pos_pred_sequence.numpy()
    ori_pred_sequence_array = ori_pred_sequence.numpy()
    pos_x_hat, pos_y_hat, pos_z_hat = [pos_pred_sequence_array[:, i] for i in [0, 1, 2]]
    ori_a_hat, ori_b_hat, ori_c_hat = [ori_pred_sequence_array[:, i] for i in [0, 1, 2]]

    pos_true_sequence = true_sequence[0, :3, :].transpose(0, 1)
    ori_true_sequence = true_sequence[0, 3:, :].transpose(0, 1)
    ori_true_sequence = six_dof_to_euler(ori_true_sequence)    
    pos_true_sequence_array = pos_true_sequence.numpy()
    ori_true_sequence_array = ori_true_sequence.numpy()
    pos_x, pos_y, pos_z = [pos_true_sequence_array[:, i] for i in [0, 1, 2]]
    ori_a, ori_b, ori_c = [ori_true_sequence_array[:, i] for i in [0, 1, 2]]

    time_axis = range(pred_sequence.shape[-1])
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(time_axis, pos_x_hat, 'r--', label='x_hat')
    plt.plot(time_axis, pos_y_hat, 'g--', label='y_hat')
    plt.plot(time_axis, pos_z_hat, 'b--', label='z_hat')
    plt.plot(time_axis, pos_x, 'r-', label='x')
    plt.plot(time_axis, pos_y, 'g-', label='y')
    plt.plot(time_axis, pos_z, 'b-', label='z')
    ax_pos = plt.gca()
    ax_pos.set_ylim([-0.5, 0.5])
    plt.legend(fontsize=8)
    
    plt.subplot(1, 2, 2)
    plt.plot(time_axis, ori_a_hat, 'r--', label='alpha_hat')
    plt.plot(time_axis, ori_b_hat, 'g--', label='beta_hat')
    plt.plot(time_axis, ori_c_hat, 'b--', label='gamma_hat')
    plt.plot(time_axis, ori_a, 'r-', label='alpha')
    plt.plot(time_axis, ori_b, 'g-', label='beta')
    plt.plot(time_axis, ori_c, 'b-', label='gamma')
    ax_ori = plt.gca()
    ax_ori.set_ylim([-torch.pi, torch.pi])
    plt.legend(fontsize=8)
    
    if return_fig:
        return fig, ax
    else:
        plt.show()


def select_scheduler(optimizer, lr_params):
    if lr_params['scheduler_type'] == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **lr_params['multistep'])
    elif lr_params['scheduler_type'] == 'cosine':
        return CosineAnnealingWarmupRestarts(optimizer, **lr_params['cosine'])
    elif lr_params['scheduler_type'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **lr_params['onecycle'])
        if scheduler.last_epoch != -1:  # not sure if necessary
            lr_params['onecycle']['last_epoch'] = scheduler.last_epoch
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **lr_params['onecycle'])
        return scheduler
