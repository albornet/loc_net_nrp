import torch
import torch.utils.data as data
import random
import pandas as pd
import torchvision.transforms as T


class CapacityDataset(data.Dataset):

    def __init__(self, csv_read, from_to, n_samples, img_dims, n_frames, n_frames_max, transform=None):

        # Parameters pre-initialization
        super(CapacityDataset, self).__init__()
        self.csv_read = csv_read
        self.h = img_dims[0]
        self.w = img_dims[1]
        self.n_frames = n_frames
        self.n_frames_max = n_frames_max
        self.speed_factor = 1
        self.transform = transform
        self.from_to = from_to
        try:
            self.dataset_length = from_to[1] - from_to[0]
        except TypeError:
            self.dataset_length = n_samples - from_to[0]

    def __getitem__(self, index):

        index += self.from_to[0]
        first_row = index * self.n_frames_max - int(index != 0)
        first_row = random.randint(first_row, first_row + self.n_frames_max - self.speed_factor * self.n_frames)
        last_row = first_row + self.speed_factor * self.n_frames
        sample_dataframe = self.csv_read.iloc[first_row:last_row:self.speed_factor, :].values
        sample_tensor = torch.tensor(sample_dataframe, dtype=torch.float)
        
        labels_pos = sample_tensor[:, 2:5]
        labels_ori = sample_tensor[:, 5:8]
        labels_ori = euler_to_six_dof(labels_ori)
        labels = torch.cat((labels_pos, labels_ori), dim=-1)
        labels = labels.permute((1, 0))  # frame index last

        images = sample_tensor[:, 8:8 + self.h * self.w].view((self.n_frames, self.h, self.w))
        images = images.permute((1, 2, 0)).view(1, self.h, self.w, self.n_frames)
        images = images * 300  # distribution of input values????
        range_shift = 7.5  # to adapt the range of input
        images = (range_shift + torch.log(torch.exp(torch.tensor(-range_shift)) + images)) / range_shift

        if self.transform:
            images = self.transform(images)
        return images, labels

    def __len__(self):
        return self.dataset_length


def get_capacity_dataloaders(dataset_path, dataset_dir, tr_ratio, n_samples,
                             n_frames, batch_size_train, batch_size_valid):

    # Initialize dataset-dependent parameters
    dataset_path = dataset_path[dataset_dir]
    n_samples = n_samples[dataset_dir]
    n_frames = n_frames[dataset_dir]  # TODO: check with n_frames_max

    # Initialize dataset file
    img_dims = (32, 32) if any([s in dataset_path for s in ['03.', '04.', '05.', '06.']]) else (2, 4)
    n_frames_max = 30 if any([s in dataset_path for s in ['05.', '06.', 'test_rot.']]) else 1001
    dataset_read = pd.read_pickle(dataset_path)

    # Training dataloader
    train_bounds = (0, int(n_samples * tr_ratio))
    train_dataset = CapacityDataset(dataset_read, train_bounds, n_samples, img_dims, n_frames, n_frames_max)
    train_dataloader = data.DataLoader(train_dataset,
        batch_size=batch_size_train, shuffle=True, sampler=None, batch_sampler=None,
        num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0,
        worker_init_fn=None, prefetch_factor=2, persistent_workers=False)

    # Validation dataloader (here, augmentation is False)
    valid_bounds = (int(n_samples * tr_ratio), None)  # None means very last one
    valid_dataset = CapacityDataset(dataset_read, valid_bounds, n_samples, img_dims, n_frames, n_frames_max)
    valid_dataloader = data.DataLoader(valid_dataset,
        batch_size=batch_size_valid, shuffle=True, sampler=None, batch_sampler=None,
        num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0,
        worker_init_fn=None, prefetch_factor=2, persistent_workers=False)

    # Return the dataloaders to the computer
    return train_dataloader, valid_dataloader


from scipy.spatial.transform import Rotation as R
def euler_to_six_dof(euler_angles):
    '''
    Transform a sequence of (discontinuous) euler angles
        into a sequence of (continuous) 6 dof angles
        ref: https://zhengyiluo.github.io/assets/pdf/Rotation_DL.pdf
    Argument: 
    - euler_angles: torch tensor of shape (n_frames, 3)
    Return:
    - torch tensor of shape (n_frames, 6)
    '''
    matrices_so3 = R.from_euler('XYZ', euler_angles).as_matrix()
    matrices_six_dof = [matrix_so3[:-1, :].transpose() for matrix_so3 in matrices_so3]  # remove last column
    matrices_six_dof = torch.stack([torch.tensor(m) for m in matrices_six_dof], dim=0)
    return matrices_six_dof.view((matrices_six_dof.shape[0], 6))

def six_dof_to_euler(six_dof_angles):
    '''
    Transform a sequence of (continuous) 6 dof angles
        into a sequence of (discontinuous) euler angles
        ref: https://zhengyiluo.github.io/assets/pdf/Rotation_DL.pdf
    Argument:
    - six_dof_angles: torch tensor of shape (n_frames, 6)
    Return:
    - torch tensor of shape (n_frames, 3)
    '''
    six_dof_angles = six_dof_angles.view(six_dof_angles.shape[0], 3, 2)
    matrices_so3 = []
    for six_dof_angle in six_dof_angles:  # shape: (3, 2)
        a_1 = six_dof_angle[:, 0]
        a_2 = six_dof_angle[:, 1]
        b_1 = a_1 / torch.norm(a_1)
        b_2 = a_2 - torch.dot(b_1, a_2) * b_1
        b_2 = b_2 / torch.norm(b_2)
        b_3 = torch.cross(b_1, b_2)
        matrices_so3.append(torch.stack([b_1, b_2, b_3], dim=0))
    matrices_so3 = torch.stack(matrices_so3, dim=0)
    euler_angles = R.from_matrix(matrices_so3).as_euler('XYZ')
    return torch.tensor(euler_angles, dtype=torch.float32)
