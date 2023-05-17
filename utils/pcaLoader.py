from scipy.spatial.transform import Rotation
from sklearn.decomposition import KernelPCA, PCA
import numpy as np

def quat2mat(quaternion):
    w, x, y, z = quaternion
    return np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                     [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                     [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])

def rotate(features, v, u, t):
    # 计算旋转轴
    axis = np.cross(v, u)
    axis /= np.linalg.norm(axis)  # 归一化旋转轴

    # 计算旋转角度
    cos_theta = np.clip(np.dot(v, u), -1, 1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    angle = np.arctan2(sin_theta, cos_theta)

    # 构造旋转矩阵
    rotation_matrix = np.eye(3) + np.sin(angle) * np.array([[0, -axis[2], axis[1]],
                                                            [axis[2], 0, -axis[0]],
                                                            [-axis[1], axis[0], 0]]) + \
                       (1 - np.cos(angle)) * np.outer(axis, axis)

    # 对特征点数组进行旋转
    t = t.reshape((3,))
    rotated_features = np.dot(features-t, rotation_matrix.T)+t

    return rotated_features


def rotPCALoader_ (channelsBatch, cam_orient, cam_trans):
    for i, channels in enumerate(channelsBatch):
        channels = channels.astype(np.float32).reshape((16, 3))
        channels_3 = channels
        # I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
        # J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        # I = np.array([0, 0, 0, 7, 8, 8, 8])  # start points
        # J = np.array([1, 4, 7, 8, 9, 10, 13])
        # 10， 11， 12， 13， 14， 15， 4， 5， 6， 1， 2， 3
        # I = np.array([1, 2, 4, 5, 10, 11, 13, 14])  # start points
        # J = np.array([2, 3, 5, 6, 11, 12, 14, 15])
        # for i in range(8):
        # #     temp = 0.25 * channels_3[I[i]] + 0.75 * channels_3[J[i]]
        # #     channels_3 = np.append(channels_3, temp.reshape((1, 3)), axis=0)
        #     temp = 0.5 * channels_3[I[i]] + 0.5 * channels_3[J[i]]
        #     channels_3 = np.append(channels_3, temp.reshape((1, 3)), axis=0)
        #     temp = 0.75 * channels_3[I[i]] + 0.25 * channels_3[J[i]]
        #     channels_3 = np.append(channels_3, temp.reshape((1, 3)), axis=0)

         # 16 * 3 keypoints of human joints
        mean = np.mean(channels_3, axis=0)
        std = np.std(channels_3, axis=0)
        channels_3 = (channels_3-mean)/std
        #print(channels_3)
        #channels_3[:, 1] = 0
        # pca = PCA(n_components=1)
        # pca.fit(channels_3)
        # ang = pca.components_[-1].reshape((3,))
        # cov_matrix = np.cov(channels_3.T)
        # val, vec = np.linalg.eig(cov_matrix)
        # ang = vec[:, np.argmax(val)]
        _, _, Vt = np.linalg.svd(channels_3)
        ang = Vt[1]
        orient = ang / np.linalg.norm(ang)

        quaternion = np.array(cam_orient).reshape((4,))
        #quaternion /= np.linalg.norm(quaternion)
        angNorm = quaternion[1:].reshape((3,))
        angNorm /= np.linalg.norm(angNorm)
        angNorm = angNorm*(-1)

        crossProd = np.cross(angNorm, orient)

        dotProd = np.dot(angNorm, orient)
        crossMat = np.array([[0, -crossProd[2], crossProd[1]],
                             [crossProd[2], 0, -crossProd[0]],
                             [-crossProd[1], crossProd[0], 0]])
        rotMat = np.eye(3) + crossMat + np.dot(crossMat, crossMat) / (1 + dotProd)


        cam_trans = cam_trans.reshape((3, ))
        res = np.dot(channels-cam_trans, rotMat.T)+cam_trans
        channelsBatch[i] = res

    return channelsBatch


def rotPCALoader (channelsBatch, cam_orient, cam_trans):
    for i, channels in enumerate(channelsBatch):
        channels = channels.astype(np.float32).reshape((16, 3))
        channels_3 = channels
        #channels_3 = np.delete(channels, [12, 15, 6, 3, 5, 2, 11, 14], axis=0)
        # I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
        # J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        #I = np.array([0, 0, 0, 7, 8, 8, 8])  # start points
        #J = np.array([1, 4, 7, 8, 9, 10, 13])
        # 10， 11， 12， 13， 14， 15， 4， 5， 6， 1， 2， 3
        # I = np.array([1, 2, 4, 5, 10, 11, 13, 14])  # start points
        # J = np.array([2, 3, 5, 6, 11, 12, 14, 15])
        # for i in range(7):
        # # #     temp = 0.25 * channels_3[I[i]] + 0.75 * channels_3[J[i]]
        # # #     channels_3 = np.append(channels_3, temp.reshape((1, 3)), axis=0)
        #     temp = 0.5 * channels_3[I[i]] + 0.5 * channels_3[J[i]]
        #     channels_3 = np.append(channels_3, temp.reshape((1, 3)), axis=0)
            # temp = 0.75 * channels_3[I[i]] + 0.25 * channels_3[J[i]]
            # channels_3 = np.append(channels_3, temp.reshape((1, 3)), axis=0)

         # 16 * 3 keypoints of human joints


        mean = np.mean(channels_3, axis=0)
        std = np.std(channels_3, axis=0)
        channels_3 = (channels_3-mean)/std
        pca = PCA(n_components=2)
        pca.fit(channels_3)
        ang0, ang1 = pca.components_[0], pca.components_[1]
        ang = np.cross(ang1, ang0)
        orient = ang / np.linalg.norm(ang)

        quaternion = np.array(cam_orient).reshape((4,))
        #angNorm = quaternion[1:].reshape((3,))
        #angNorm /= np.linalg.norm(angNorm)

        R = quat2mat(quaternion)
        R = np.dot(R, np.array([0, 0, -1]))
        R /= np.linalg.norm(R)
        res = rotate(channels, orient, R, cam_trans)
        res = (res-np.mean(res, axis=0)) / np.std(res, axis=0)
        res = res*std + mean
        #cam_trans = cam_trans.reshape((3, ))
        #res = np.dot(channels-cam_trans, rotMat.T)+cam_trans
        channelsBatch[i] = res

    return channelsBatch
