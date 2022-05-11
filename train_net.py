import torch
import os
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from src.model import PredNet
from src.utils import train_fn, valid_fn
from src.dataset_fn_multi import get_capacity_dataloaders

# Model parameters
load_model, do_time_aligned, n_layers = True, True, 3
do_untouched_bu = False
batch_size_train, batch_size_valid = 4, 64
pos_layers = tuple([l for l in [1, 2] if l < n_layers])  # set as [] for not using it
ori_layers = tuple([l for l in [1, 2] if l < n_layers])  # set as [] for not using it
bu_channels = (64, 128, 256, 512)[:n_layers]
td_channels = (64, 128, 256, 512)[:n_layers]
td_layers = ('H', 'H', 'H', 'H')[:n_layers]  # 'Hgru', 'Illusory', 'Lstm', 'Conv'
dropout_rates = (0.0, 0.0, 0.0, 0.0)[:n_layers]
device = 'cuda'  # 'cuda', 'cpu'

# Training parameters
n_epochs_run, n_epochs_save, epoch_to_load = 1000, 10, None
learning_rate, lr_decay_time, lr_decay_rate, betas = 1e-4, 50, 0.75, (0.9, 0.98)
first_cycle_steps, cycle_mult, max_lr, min_lr, warmup_steps, gamma = 10, 1.0, 1e-4, 1e-5, 2, 1.0
scheduler_type = 'multistep'  # 'multistep', 'cosannealwarmuprestart'
loss_w = {
    'latent': (5.0, 0.0, 0.0, 0.0)[:n_layers],  # now this is image loss
    'loc_mae': 1.0 if len(pos_layers) > 0 else 0.0,
    'loc_mse': 0.0 if len(pos_layers) > 0 else 0.0,
    'loc_bce': 0.0 if len(pos_layers) > 0 else 0.0}
do_localization = not (len(pos_layers) + len(ori_layers) == 0 or \
    sum([loss_w['loc_' + k] for k in ['mae', 'mse', 'bce']]) == 0)

# Build model name
model_name = \
      f'TA{int(do_time_aligned)}_BU{bu_channels}_TD{td_channels}_TL{td_layers}'\
      f'_PL{pos_layers}_OL{ori_layers}_DR{tuple([int(10 * r) for r in dropout_rates])}'
model_name = model_name.replace('.', '-').replace(',', '-').replace(' ', '').replace("'", '')

# Dataset
dataset_dir = r'D:\DL\datasets\nrp'
dataset_type = 'capa'  # 'capa', 'multi'  TODO: check with Michael why 'multi' does not work
n_frames, n_backprop_frames, t_start = 30, 5, n_layers
augmentation, remove_ground, tr_ratio = True, True, 0.8
n_labels = 9  # 3d-location and 6d-orientation (continuous angles)
if dataset_type == 'multi':
    dataset_path = os.path.join(dataset_dir, 'multi_sensory', 'capacity_dataset.pkl')
    n_samples = 1500  # ???
elif dataset_type == 'capa':
    dataset_path = os.path.join(dataset_dir, 'capacity', 'capacity_dataset_06.pkl')
    n_samples = 560
train_dl, valid_dl = get_capacity_dataloaders(dataset_path,
                                              tr_ratio,
                                              n_samples,
                                              n_frames,
                                              batch_size_train,
                                              batch_size_valid)

# Load the model
if not load_model:
    print(f'\nCreating model: {model_name}')
    model = PredNet(model_name,
                    n_layers,
                    td_layers,
                    pos_layers,
                    ori_layers,
                    bu_channels,
                    td_channels,
                    do_localization,
                    device)
    train_losses, valid_losses, last_epoch = [], [], 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas)
    if scheduler_type == 'multistep':
        milestones = range(lr_decay_time,10 * (n_epochs_run + 1), lr_decay_time)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones,
                                                         lr_decay_rate)
    elif scheduler_type == 'cosannealwarmuprestart':
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  first_cycle_steps=first_cycle_steps,
                                                  cycle_mult=cycle_mult,
                                                  max_lr=max_lr,
                                                  min_lr=min_lr,
                                                  warmup_steps=warmup_steps,
                                                  gamma=gamma)
    train_losses, valid_losses = [], []
else:
    print(f'\nLoading model: {model_name}')
    lr_params = [scheduler_type, learning_rate, lr_decay_time, lr_decay_rate, betas,
                 first_cycle_steps, cycle_mult, max_lr, min_lr, warmup_steps, gamma, betas]
    model, optimizer, scheduler, train_losses, valid_losses = PredNet.load_model(model_name,
                                                                                 n_epochs_run,
                                                                                 epoch_to_load,
                                                                                 lr_params)
    last_epoch = scheduler.last_epoch

# Train the network
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Training network ({n_params} trainable parameters)')
for epoch in range(last_epoch, last_epoch + n_epochs_run):
    print(f'\nRunning epoch {epoch}:')
    train_losses.extend(train_fn(train_dl, model, optimizer, loss_w,
                                 t_start, n_backprop_frames, epoch))
    valid_losses.extend(valid_fn(valid_dl, model, loss_w, t_start, epoch))
    scheduler.step()
    if (epoch + 1) % n_epochs_save == 0:
        model.save_model(optimizer, scheduler, train_losses, valid_losses)
