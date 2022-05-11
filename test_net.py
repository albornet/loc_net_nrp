import torch
import numpy as np
import imageio
from src.model import PredNet

def load_image_sequence(folder_path, n_frames):
    ''' Loads a sequence of images from a given folder path.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing the images.
    n_frames : int
        Number of frames to load.
    
    Returns
    -------
    images : np.array
        Array of shape (batch_size, 3, height, width, n_frames)
        containing the images.
    '''
    list_of_torch_tensors = torch.rand(1, 3, 256, 256, 30)  # TODO: load from folder_path
    list_of_torch_tensors = list_of_torch_tensors[..., :n_frames]
    return list_of_torch_tensors  # of shape (1, n_channels, width, height)

def plot_seg_image_sequence(seg_image_sequence):
    ''' Plots a sequence of segmentation images.
    
    Parameters
    ----------
    seg_image_sequence : np.array
        Array of shape (batch_size, 3, height, width, n_frames)
        containing the segmentation images.
    
    Returns
    -------
    None
    '''
    n_frames = len(seg_image_sequence)
    seg_image_stack = np.stack(seg_image_sequence, axis=-1)
    seg_image_rgb = onehot_to_rgb(seg_image_stack).transpose(0, 2, 3, 1, 4)
    seg_image_rgb = (255 * seg_image_rgb).astype(np.uint8)
    for sample_idx, seg_sample in enumerate(seg_image_rgb):
        seg_sample_list = [seg_sample[..., t] for t in range(n_frames)]
        imageio.mimsave(f'./seg_output_{sample_idx:02}.gif',
                        seg_sample_list,
                        duration=0.1)

def onehot_to_rgb(onehot_array):
    ''' Converts a one-hot encoded array to an RGB array.
    
    Parameters
    ----------
    onehot_array : np.array
        Array of shape (batch_size, height, width, n_frames, n_classes)
        containing the one-hot encoded images.

    Returns
    -------
    rgb_array : np.array
        Array of shape (batch_size, height, width, n_frames, 3)
        containing the RGB images.
    '''
    batch_size, num_classes, w, h, n_frames = onehot_array.shape
    rgb_array = np.zeros((batch_size, 3, w, h, n_frames))
    hue_space = np.linspace(0.0, 1.0, num_classes + 1)[:-1]
    rgb_space = [hsv_to_rgb(hue) for hue in hue_space]
    for n in range(num_classes):
        class_tensor = onehot_array[:, n]
        for c, color in enumerate(rgb_space[n]):
            rgb_array[:, c] += color * class_tensor
    return rgb_array

def hsv_to_rgb(hue):
    ''' Converts a hue value to an RGB color.
    
    Parameters
    ----------
    hue : float
        Hue value in the range [0.0, 1.0].

    Returns
    -------
    rgb : np.array
        Array of shape (3,) containing the RGB color.
    '''
    v = 1 - abs((int(hue * 360) / 60) % 2 - 1)
    hsv_space = [
        [1, v, 0], [v, 1, 0], [0, 1, v],
        [0, v, 1], [v, 0, 1], [1, 0, v]]
    return hsv_space[int(hue * 6)]

if __name__ == '__main__':
    ''' Test the PredNet model on a sequence of images.'''

    # Load the model
    device = 'cuda'  # or 'cpu'
    model = PredNet(model_name='my_model',
                    n_classes=5,
                    n_layers=3,
                    seg_layers=(1, 2),
                    bu_channels=(64, 128, 256),
                    td_channels=(64, 128, 256),
                    do_segmentation=True,
                    device=device)
    model.eval()

    # Load the images
    seg_image_sequence = []
    image_sequence = load_image_sequence()
    n_frames = image_sequence.shape[-1]
    
    # Predict the segmentation
    with torch.no_grad():
        for t in range(n_frames):
            image = image_sequence[..., t].to(device=device)
            _, _, seg_image = model(image, t)
            seg_image_numpy = seg_image.cpu().numpy()
            seg_image_sequence.append(seg_image_numpy)
    
    # Plot the segmentation
    plot_seg_image_sequence(seg_image_sequence)
