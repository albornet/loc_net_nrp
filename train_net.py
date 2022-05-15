import torch
# from src.model import PredNet
from src.model_old import PredNet
from src.utils import train_fn, valid_fn, select_scheduler
from src.dataset_fn_multi import get_capacity_dataloaders

def main():

    # General parameters
    load_model = True
    n_epochs_run = 20
    n_epochs_save = 5
    epoch_to_load = None  # None to load the last available epoch

    # Model parameters
    n_layers = 3
    model_params = {
        'n_layers': n_layers,
        'td_layers': ('H', 'H', 'H', 'H', 'H', 'H')[:n_layers],  # 'Hgru', 'Illusory', 'Lstm', 'Conv'
        'img_layers': tuple([l for l in [0, 2] if l < n_layers]),  # set as [] for not using it
        'pos_layers': tuple([l for l in [0, 1] if l < n_layers]),  # set as [] for not using it
        'ori_layers': tuple([l for l in [0, 1] if l < n_layers]),  # set as [] for not using it
        'bu_channels': (64, 128, 256, 512, 1024)[:n_layers],
        'td_channels': (64, 128, 256, 512, 1024)[:n_layers],
        'device': 'cuda'}  # 'cuda', 'cpu'

    # Loss parameters
    loss_params = {
        'n_backprop_frames': 10,
        't_start': 1,
        'latent': (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)[:n_layers],
        'prd_mae': 1.0 if len(model_params['img_layers']) > 0 else 0.0,
        'loc_mae': 1.0 if len(model_params['pos_layers']) > 0 else 0.0,
        'loc_mse': 0.0 if len(model_params['pos_layers']) > 0 else 0.0,
        'loc_bce': 0.0 if len(model_params['pos_layers']) > 0 else 0.0}

    # Dataset parameters
    data_params = {
        'batch_size_train': 4,
        'batch_size_valid': 512,
        'tr_ratio': 0.8,
        'dataset_dir': 'capa', # 'capa', 'multi'
        'dataset_path': {
            'capa': r'D:\DL\datasets\nrp\capacity\capacity_dataset_06.pkl',
            'multi': r'D:\DL\datasets\nrp\multi_sensory\capacity_dataset.pkl'},
        'n_samples': { 'capa': 560, 'multi': 1500},
        'n_frames': {'capa': 30, 'multi': 30}}
    train_dl, valid_dl = get_capacity_dataloaders(**data_params)

    # Training parameters
    lr = 5e-4
    scheduler_type = 'multistep'  # 'multistep', 'cosine', 'onecycle'
    lr_params = {
        'scheduler_type': scheduler_type,
        'optimizer': {'lr': lr, 'betas': (0.9, 0.98), 'eps': 1e-8},
        'multistep': {'milestones': range(5, 10000, 5), 'gamma': 0.75},
        'cosine': {'first_cycle_steps': 10, 'cycle_mult': 1.0, 'max_lr': lr,
                'min_lr': lr / 100, 'warmup_steps': 2, 'gamma': 1.0},
        'onecycle': {'max_lr': lr, 'steps_per_epoch': len(train_dl), 'epochs': n_epochs_run}}

    # Load the model, optimizer and scheduler
    model_name = 'BU' + str(model_params['bu_channels']) + '_TD' + str(model_params['td_channels'])\
            + '_TL' + str(model_params['td_layers']) + '_IL' + str(model_params['img_layers'])\
            + '_PL' + str(model_params['pos_layers']) + '_OL' + str(model_params['ori_layers'])
    model_name = model_name.replace('.', '-').replace(',', '-').replace(' ', '').replace("'", '')
    if not load_model:
        print(f'\nCreating model: {model_name}')
        model = PredNet(model_name, **model_params)
        train_losses, valid_losses = [], []
        optimizer = torch.optim.AdamW(model.parameters(), **lr_params['optimizer'])
        scheduler = select_scheduler(optimizer, lr_params)
    else:
        print(f'\nLoading model: {model_name}')
        model, optimizer, scheduler, train_losses, valid_losses = PredNet.load_model(model_name,
                                                                                    epoch_to_load,
                                                                                    lr_params)

    # Train the network
    last_epoch = len(valid_losses) // (1 + len(valid_dl) // data_params['batch_size_valid'])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Training network ({n_params} trainable parameters)')
    for epoch in range(last_epoch, last_epoch + n_epochs_run):
        print(f'\nRunning epoch {epoch}:')
        train_losses.extend(train_fn(train_dl, model, optimizer, scheduler, loss_params, epoch))
        valid_losses.extend(valid_fn(valid_dl, model, loss_params, epoch))
        model.save_model(optimizer, scheduler, train_losses, valid_losses, epoch, n_epochs_save)

if __name__ == '__main__':
    main()