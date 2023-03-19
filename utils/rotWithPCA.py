from sklearn.decomposition import PCA
import numpy as np


def rotWithPCA (channels):
    channels_3 = np.reshape(channels, (3, -1))
    pca = PCA(n_components=1)
    ang = pca.fit_transform(channels_3)
    ang = np.reshape(ang, (3,))
    angNorm = ang / np.linalg.norm(ang)
    orient = np.array([0, 0, 1])
    crossProd = np.cross(orient, angNorm)
    dotProd = np.dot(orient, angNorm)
    crossMat = np.array([[0, -crossProd[2], crossProd[1]],
                         [crossProd[2], 0, -crossProd[0]],
                         [-crossProd[1], crossProd[0], 0]])
    rotMat = np.eye(3) + crossMat + np.dot(crossMat, crossMat) / (1 + dotProd)
    res = np.dot(channels, rotMat)
    return res