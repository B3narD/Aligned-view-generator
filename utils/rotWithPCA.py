from scipy.spatial.transform import Rotation
from sklearn.decomposition import KernelPCA, PCA
import numpy as np


def rotWithPCA (channels, bodyConstrain=False):
    channels = np.reshape(channels, (16, -1)) # 16 * 3 keypoints of human joints

    if bodyConstrain==True:
        rows = [0, 8, 10, 13, 1, 4]
        channels_3 = np.concatenate([channels[rows]], axis = 1)
        channels_3 = np.reshape(channels_3, (3, -1))

    else:
        channels_3 = np.reshape(channels, (3, -1)) # 3 * 16

    pca = PCA(n_components=3)
    pca.fit(channels_3.T)
    ang = pca.components_[0].reshape(3,) # find those keypoints' orientation by PCA
    # TODO:
    orient = ang / np.linalg.norm(ang)
    angNorm = np.array([0, 0, 1])
    crossProd = np.cross(angNorm, orient)

    dotProd = np.dot(angNorm, orient)
    crossMat = np.array([[0, -crossProd[2], crossProd[1]],
                         [crossProd[2], 0, -crossProd[0]],
                         [-crossProd[1], crossProd[0], 0]])
    rotMat = np.eye(3) + crossMat + np.dot(crossMat, crossMat) / (1 + dotProd)
    #res = np.dot(channels, rotMat)
    # rot_mat = Rotation.align_vectors(ang, np.array([0, 0, 1]).reshape(1, 3))[0]
    # res = np.dot(rot_mat.as_matrix(), channels.T).T
    # TODO:
    # rotMat = np.eye(3)
    # target = np.array([0, 1, 0])
    # if np.abs(orient[0, 2]) > 0.01:
    #     v = np.cross(orient, target)
    #     c = np.dot(orient, target)
    #     s = np.sqrt(1 - c * c)
    #     vx, vy, vz = v[0, 0], v[0, 1], v[0, 2]
    #     rotMat = np.array([[vx * vx * (1 - c) + c, vx * vy * (1 - c) - vz * s, vx * vz * (1 - c) + vy * s],
    #                        [vx * vy * (1 - c) + vz * s, vy * vy * (1 - c) + c, vy * vz * (1 - c) - vx * s],
    #                        [vx * vz * (1 - c) - vy * s, vy * vz * (1 - c) + vx * s, vz * vz * (1 - c) + c]])

    res = np.dot(channels, rotMat)

    #return res, orient
    return res, orient


# def rotBatch (channels):
#     channels = channels.cpu().numpy()
#     print(channels.shape)
#     for i, channel in enumerate(channels):
#         channels[i] = channels[]