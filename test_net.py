import torch
import matplotlib.pyplot as plt
# from src.model import PredNet
from src.model_old import PredNet
from src.dataset_fn_multi import six_dof_to_euler, get_capacity_dataloaders


def main():
    ''' Test the PredNet model on a sequence of images.'''
    
    # Load model
    device = 'cuda'
    model_name = 'BU(64-128-256)_TD(64-128-256)_TL(H-H-H)_IL(0-2)_PL(0-1)_OL(0-1)'
    model, _, _, _, _ = PredNet.load_model(model_name)
    model.eval()
    model.to(device)

    # Load dataset
    data_params = {
        'batch_size_train': 1,
        'batch_size_valid': 1,
        'tr_ratio': 0.8,
        'dataset_dir': 'capa', # 'capa', 'multi'
        'dataset_path': {
            'capa': r'D:\DL\datasets\nrp\capacity\capacity_dataset_06.pkl',
            'multi': r'D:\DL\datasets\nrp\multi_sensory\capacity_dataset.pkl'},
        'n_samples': { 'capa': 560, 'multi': 1500},
        'n_frames': {'capa': 30, 'multi': 30}}
    _, valid_dl = get_capacity_dataloaders(**data_params)
    
    # Run the model and plot the output prediction
    with torch.no_grad():
        for (input_sequence, label_sequence) in valid_dl:
            loc_tensor_sequence = torch.zeros(label_sequence.shape, device=device)
            for t in range(input_sequence.shape[-1]):
                input_ = input_sequence[..., t].to(device=device)
                _, _, loc_tensor = model(input_, t)
                loc_tensor_sequence[..., t] = loc_tensor
            pred_sequence = loc_tensor_sequence.detach().cpu()
            true_sequence = label_sequence
            plot_loc_sequence(pred_sequence, true_sequence)


def plot_loc_sequence(pred_sequence, true_sequence):
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
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(time_axis, pos_x_hat, 'r--', label='x_hat')
    plt.plot(time_axis, pos_y_hat, 'g--', label='y_hat')
    plt.plot(time_axis, pos_z_hat, 'b--', label='z_hat')
    plt.plot(time_axis, pos_x, 'r-', label='x')
    plt.plot(time_axis, pos_y, 'g-', label='y')
    plt.plot(time_axis, pos_z, 'b-', label='z')
    plt.legend(fontsize=8)
    
    plt.subplot(1, 2, 2)
    plt.plot(time_axis, ori_a_hat, 'r--', label='alpha_hat')
    plt.plot(time_axis, ori_b_hat, 'g--', label='beta_hat')
    plt.plot(time_axis, ori_c_hat, 'b--', label='gamma_hat')
    plt.plot(time_axis, ori_a, 'r-', label='alpha')
    plt.plot(time_axis, ori_b, 'g-', label='beta')
    plt.plot(time_axis, ori_c, 'b-', label='gamma')
    ax = plt.gca()
    ax.set_ylim([-torch.pi, torch.pi])
    plt.legend(fontsize=8)
    plt.show()
            

if __name__ == '__main__':
    main()