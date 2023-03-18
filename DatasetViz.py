import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from common.data_loader import PoseDataSet
from common.h36m_dataset import Human36mDataset
from common.viz import show3Dpose, show2Dpose
from common.camera import project_to_2d
from utils.data_utils import fetch, create_2d_data, read_3d_data
# common and utils can be reused
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        batch_size = 16
    else:
        batch_size = 128

    dataset_path = os.path.join('data', 'data_3d_' + 'h36m' + '.npz')
    dataset = Human36mDataset(dataset_path)
    dataset = read_3d_data(dataset)
    subjects_train = ['S1', 'S5', 'S6', 'S7', 'S8']
    keypoints = create_2d_data(os.path.join('data', 'data_2d_' + 'h36m' + '_' + 'gt' + '.npz'), dataset)

    poses_train, poses_train_2d, actions_train, cams_train = fetch(subjects_train, dataset, keypoints)
    train_gt2d3d_loader = DataLoader(PoseDataSet(poses_train, poses_train_2d, actions_train, cams_train),
                                     batch_size=batch_size,
                                     shuffle=True, num_workers=4, pin_memory=True)

    for i, (inputs_3d, _, _, cam_param) in enumerate(train_gt2d3d_loader):
        inputs_3d, cam_param = inputs_3d.to(device), cam_param.to(device)
        inputs_2d = project_to_2d(inputs_3d, cam_param)

        fig3d = plt.figure(figsize=(16, 8))

        # input 3D
        ax3din = fig3d.add_subplot(1, 2, 1, projection='3d')
        ax3din.set_title('input 3D')
        show3Dpose(inputs_3d.cpu().detach().numpy()[0], ax3din, gt=True)

        ax2din = fig3d.add_subplot(1, 2, 2)
        ax2din.set_title('input 2d')
        show2Dpose(inputs_2d.cpu().detach().numpy()[0], ax2din)

        os.makedirs("./data_viz", exist_ok=True)
        img_name = "./data_viz/sample{}".format(i + 1)
        plt.savefig(img_name)
        plt.close("all")
