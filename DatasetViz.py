import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from common.data_loader import PoseDataSet
from common.h36m_dataset import Human36mDataset
from common.viz import show3Dpose, show2Dpose
from common.camera import project_to_2d, project_to_2d_linear
from utils.data_utils import fetch, create_2d_data, read_3d_data
from utils.rotWithPCA import rotWithPCA

# common and utils can be reused
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        batch_size = 16
    else:
        batch_size = 128

    dataset_path = os.path.join('data', 'data_3d_' + 'h36m' + '.npz')
    dataset_path_2 = os.path.join('data', 'data_3d_' + 'h36m_' + 'PCA' + '.npz')
    dataset = Human36mDataset(dataset_path)
    #dataset_2 = Human36mDataset(dataset_path)
    dataset = read_3d_data(dataset)
    #dataset_2 = read_3d_data(dataset_2)
    subjects_train = ['S1', 'S5', 'S6', 'S7', 'S8']
    keypoints = create_2d_data(os.path.join('data', 'data_2d_' + 'h36m' + '_' + 'gt' + '.npz'), dataset)
    keypoints_2 = create_2d_data(os.path.join('data', 'data_2d_' + 'h36m' + '_' + 'gt_PCA' + '.npz'), dataset)

    poses_train, poses_train_2d, actions_train, cams_train = fetch(subjects_train, dataset, keypoints)
    poses_train_2, poses_train_2d_2, actions_train_2, cams_train_2 = fetch(subjects_train, dataset, keypoints_2)
    train_gt2d3d_loader = DataLoader(PoseDataSet(poses_train, poses_train_2d, actions_train, cams_train),
                                     batch_size=batch_size,
                                     shuffle=False, num_workers=2, pin_memory=False)
    train_gt2d3d_loader_2 = DataLoader(PoseDataSet(poses_train_2, poses_train_2d_2, actions_train_2, cams_train_2),
                                       batch_size=batch_size,
                                       shuffle=False, num_workers=2, pin_memory=False)
    for (i, (inputs_3d, inputs_2d, _, cam_param)), (inputs_3d_PCA, inputs_2d_PCA, _, cam_param_PCA) in zip(enumerate(train_gt2d3d_loader),
                                                                                     train_gt2d3d_loader_2):
        inputs_3d, cam_param = inputs_3d.to(device), cam_param.to(device)
        inputs_3d_PCA, cam_param_PCA = inputs_3d_PCA.to(device), cam_param_PCA.to(device)
        fig3d = plt.figure(figsize=(16, 8))

        ax3din1 = fig3d.add_subplot(1, 4, 1, projection='3d')
        show3Dpose(inputs_3d.cpu().detach().numpy()[0], ax3din1, gt=True)

        ax3din2 = fig3d.add_subplot(1, 4, 2, projection='3d')
        show3Dpose(inputs_3d_PCA.cpu().detach().numpy()[0], ax3din2, gt=True)

        inputs_2d = inputs_2d.to(device)
        inputs_2d_PCA = inputs_2d_PCA.to(device)

        ax2din1 = fig3d.add_subplot(1, 4, 3)
        ax2din1.set_title('input 2d')
        show2Dpose(inputs_2d.cpu().detach().numpy()[0], ax2din1)

        ax2din2 = fig3d.add_subplot(1, 4, 4)
        ax2din2.set_title('input 2d')
        show2Dpose(inputs_2d_PCA.cpu().detach().numpy()[0], ax2din2)

        os.makedirs("./data_viz", exist_ok=True)
        img_name = "./data_viz/sample{}".format(i + 1)
        plt.savefig(img_name)
        plt.close("all")
        if i == 50:
            break
    # for i, (inputs_3d, _, _, cam_param) in enumerate(train_gt2d3d_loader):
    #
    #     inputs_3d, cam_param = inputs_3d.to(device), cam_param.to(device)
    #     # TODO: make rotation work on batch
    #     rot_3d, orient = rotWithPCA(inputs_3d.cpu().detach().numpy()[0], bodyConstrain=False)
    #     fig3d = plt.figure(figsize=(16, 8))
    #
    #     # input 3D
    #     ax3din1 = fig3d.add_subplot(1, 4, 1, projection='3d')
    #     ax3din1.set_title('input 3D')
    #     show3Dpose(inputs_3d.cpu().detach().numpy()[0], ax3din1, gt=True)
    #
    #     ax3din2 = fig3d.add_subplot(1, 4, 2, projection='3d')
    #     ax3din2.set_title('rotate 3D')
    #     show3Dpose(rot_3d, ax3din2, pred=True)
    #
    #     inputs_2d = project_to_2d_linear(inputs_3d, cam_param)
    #     rot_2d = project_to_2d_linear(torch.tensor(rot_3d.reshape(1, 16, -1)).to(device), cam_param[0].reshape(1, -1))
    #     # inputs_2d = project_to_2d(inputs_3d, cam_param)
    #     # rot_2d = project_to_2d(torch.tensor(rot_3d.reshape(1, 16, -1)).to(device), cam_param[0].reshape(1, -1))
    #
    #
    #     ax2din1 = fig3d.add_subplot(1, 4, 3)
    #     ax2din1.set_title('input 2d')
    #     show2Dpose(inputs_2d.cpu().detach().numpy()[0], ax2din1)
    #
    #     ax2din2 = fig3d.add_subplot(1, 4, 4)
    #     ax2din2.set_title("rotate 2d")
    #     show2Dpose(rot_2d.cpu().detach().numpy()[0], ax2din2)
    #
    #     os.makedirs("./data_viz", exist_ok=True)
    #     img_name = "./data_viz/sample{}".format(i + 1)
    #     plt.savefig(img_name)
    #     plt.close("all")
    #     if i == 50:
    #         break
