from scipy.spatial.transform import Rotation
from sklearn.decomposition import KernelPCA, PCA
import numpy as np


def rotPCALoader (channelsBatch, cam_orient):
    for i, channels in enumerate(channelsBatch):
        channels = channels.astype(np.float32)
        channels_3 = np.reshape(channels, (16, 3)) # 16 * 3 keypoints of human joints
        #channels_3[-1, :] = 0
        pca = PCA(n_components=1)
        pca.fit(channels_3)
        ang = pca.components_[-1].reshape((3,))
        #print(ang.shape)

        orient = ang / np.linalg.norm(ang)
        quaternion = np.array(cam_orient).reshape((4, ))
        quaternion /= np.linalg.norm(quaternion)
        angNorm = quaternion[1:].reshape((3,))
        crossProd = np.cross(angNorm, orient)

        dotProd = np.dot(angNorm, orient)
        crossMat = np.array([[0, -crossProd[2], crossProd[1]],
                             [crossProd[2], 0, -crossProd[0]],
                             [-crossProd[1], crossProd[0], 0]])
        rotMat = np.eye(3) + crossMat + np.dot(crossMat, crossMat) / (1 + dotProd)

        res = np.dot(channels, rotMat)
        channelsBatch[i] = res

    return channelsBatch
