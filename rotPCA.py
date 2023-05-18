import numpy as np
from common.camera import world_to_camera, project_to_2d, image_coordinates
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation
from utils.utils import wrap
import copy
from common.camera import normalize_screen_coordinates
import matplotlib.pyplot as plt
from common.viz import show3Dpose
from mpl_toolkits.mplot3d import Axes3D


data_3d_doc = np.load('data_3d_h36m.npz', allow_pickle=True)
data_2d_doc = np.load('data_2d_h36m_gt.npz', allow_pickle = True)

#  定义虚拟相机，内参按照相机1制定，外参方向和移动均暂不指定
virtual_camera_intrinsic = {
        'id': '54132323',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70,  # Only used for visualization
    }

virtual_camera_extrinsic = {
            'orientation': [],
            'translation': [],
        }
# 将虚拟相机内外参结合起来，按照Human36mDataset中读取数据的方法将内外参字典合并并转换成统一格式
# 转换包括：将除了id, res_w, res_h外其余所有项变为array格式，将画幅按照中心成像位置进行校准，校准焦距
# 将translation中的mm单位换算成米（此时dict中translation为空，需在后续转换）
cam_virtual = {**virtual_camera_intrinsic, **virtual_camera_extrinsic}
cam_copy= copy.deepcopy(cam_virtual)
for k,v in cam_copy.items():
    if k not in ['id', 'res_w', 'res_h']:
        cam_copy[k] = np.array(v, dtype='float32')
cam_copy['center'] = normalize_screen_coordinates(cam_copy['center'],
            w=cam_copy['res_w'], h=cam_copy['res_h']).astype('float32')
cam_copy['focal_length'] = cam_copy['focal_length'] / cam_copy['res_w'] * 2.0
if 'translation' in cam_copy:
        cam_copy['translation'] = cam_copy['translation'] / 1000
cam_copy['intrinsic'] = np.concatenate((cam_copy['focal_length'],
                                                   cam_copy['center'],
                                                   cam_copy['radial_distortion'],
                                                   cam_copy['tangential_distortion']))


data_3d = data_3d_doc['positions_3d'].item()
data_2d = data_2d_doc['positions_2d'].item()
data_3d_rot = copy.deepcopy(data_3d)

# data_3d作为原数据备份，在data_3d_rot中进行pca，删除节点等工作
def delete_joints(kp3=[]):
    remove_list = [4,5,9,10,11,14,16,20,21,22,23,24,28,29,30,31]
    kp3_rot = kp3.copy()
    for i in reversed(remove_list):
        kp3_rot = np.delete(kp3_rot,i,axis=0)
    return kp3_rot
print(data_3d['S1'].keys())

# 将32个节点删除为16个节点
demo = data_3d['S1']['Sitting 2'][100]

# 读取S1 Directions文件中的第一个图像数据作为demo


# projection_to_plane函数输入三维图像数据，如demo。该函数运用pca将三维图像降低成两维，返回kp3_rotted
# 即降维后的数据(世界坐标系，对3D降维后的二维数据点集)，以及normal_vector（两个主成分向量组成平面的法向量）
def projection_to_plane(kp3_rot = []):
    pca = PCA(n_components=2)
    pca.fit(kp3_rot)
    kp3_rotted = pca.transform(kp3_rot)
    components = pca.components_
    normal_vector = np.cross(components[0], components[1])
    normal_vector = normal_vector/np.linalg.norm(normal_vector)
    return kp3_rotted, normal_vector

# 得到normal_vector后，我们需将相机转换到镜头沿normal_vector的方向。
# 注：相机初始位置于世界坐标系原点，相机位姿坐标系的建立为：z轴为沿相机镜头向前，y轴为指向相机底部, x轴指向相机左侧
# 由世界坐标系到相机初始位置也有旋转矩阵，该矩阵为R, 尚不确定相机初始位姿于世界坐标系的关系，需要进行可视化以校准
# 该函数根据normal_vector的朝向给出相机从初始位置到指定位置的旋转矩阵
def get_rot_matrix(normal_vector):
    cam_Y_axis = normal_vector
    cam_Z_axis = np.zeros_like(cam_Y_axis)
    x = cam_Y_axis[0]
    y = cam_Y_axis[1]
    z = cam_Y_axis[2]
    cam_Z_axis[0] = -x*z/np.sqrt((x**2+y**2)*(x**2+y**2+z**2))
    cam_Z_axis[1] = -y*z/np.sqrt((x**2+y**2)*(x**2+y**2+z**2))
    cam_Z_axis[2] = np.sqrt(x**2+y**2)/np.sqrt(x**2+y**2+z**2)
    cam_X_axis = np.cross(cam_Y_axis, cam_Z_axis)
    cam_X_axis = cam_X_axis/np.linalg.norm(cam_X_axis)
    cam_rot_pca = np.column_stack((cam_X_axis, cam_Z_axis, cam_Y_axis))
    # Maybe need to be changed
    cam_rot_init = np.array([[1,0,0],[0,1,0],[0,0,1]])
    R = cam_rot_pca.dot(np.linalg.inv(cam_rot_init))
    return R
def get_rot_matrix_two(normal_vector):
    cam_Y_axis = normal_vector
    cam_Z_axis = np.zeros_like(cam_Y_axis)
    x = cam_Y_axis[0]
    y = cam_Y_axis[1]
    z = cam_Y_axis[2]
    cam_Z_axis[0] = x*z/np.sqrt((x**2+y**2)*(x**2+y**2+z**2))
    cam_Z_axis[1] = y*z/np.sqrt((x**2+y**2)*(x**2+y**2+z**2))
    cam_Z_axis[2] = -np.sqrt(x**2+y**2)/np.sqrt(x**2+y**2+z**2)
    cam_X_axis = -np.cross(cam_Y_axis, cam_Z_axis)
    cam_X_axis = cam_X_axis/np.linalg.norm(cam_X_axis)
    cam_rot_pca = np.column_stack((cam_X_axis, cam_Z_axis, cam_Y_axis))
    # Maybe need to be changed
    cam_rot_init = np.array([[1,0,0],[0,1,0],[0,0,1]])
    R = cam_rot_pca.dot(np.linalg.inv(cam_rot_init))
    return R

# 根据上一函数得到的旋转矩阵，将该矩阵转换为四元数形式
def rotmatrix_to_quat(R):
    q = Rotation.from_matrix(R).as_quat()
    return q

demo_rot = delete_joints(demo)
demo_rot[:, [1, 2]] = demo_rot[:, [2, 1]]
demo_rot[:, 1] *= -1
demo_rotted,demo_vec = projection_to_plane(demo_rot)
for x in range(-5,6):
    demo_R = get_rot_matrix(demo_vec)
    demo_q = rotmatrix_to_quat(demo_R)
    cam_copy['orientation'] = np.array(demo_q, dtype='float32')
    demo_trans = 5.478*demo_vec+[0,x,0]
    cam_copy['translation'] = np.array(demo_trans, dtype='float32')
    pos_3d_demo = world_to_camera(demo_rot, R=cam_copy['orientation'], t=cam_copy['translation'])
    pos_2d_demo = wrap(project_to_2d, True, pos_3d_demo, cam_copy['intrinsic'])
    pos_2d_pixel_space_demo = image_coordinates(pos_2d_demo, w=cam_copy['res_w'], h=cam_copy['res_h'])
    pos_2d_pixel_space_demo = pos_2d_pixel_space_demo.astype('float32')
    for i in range(len(pos_2d_pixel_space_demo)):
        pos_2d_pixel_space_demo[i][0], pos_2d_pixel_space_demo[i][1] = pos_2d_pixel_space_demo[i][1], pos_2d_pixel_space_demo[i][0]
    output_2d = pos_2d_pixel_space_demo

    demo_R_2 = get_rot_matrix_two(demo_vec)
    demo_q_2 = rotmatrix_to_quat(demo_R_2)
    cam_copy_2 = cam_copy.copy()
    cam_copy_2['orientation'] = np.array(demo_q_2, dtype='float32')
    demo_trans_2 = 5.478*demo_vec+[0,x,0]
    cam_copy_2['translation'] = np.array(demo_trans_2, dtype='float32')
    pos_3d_demo_2 = world_to_camera(demo_rot, R=cam_copy_2['orientation'], t=cam_copy_2['translation'])
    pos_2d_demo_2 = wrap(project_to_2d, True, pos_3d_demo_2, cam_copy_2['intrinsic'])
    pos_2d_pixel_space_demo_2 = image_coordinates(pos_2d_demo_2, w=cam_copy_2['res_w'], h=cam_copy_2['res_h'])
    output_2d_2= pos_2d_pixel_space_demo_2.astype('float32')
    # output_2d_2 = swap_columns(pos_2d_pixel_space_demo)
    if(210<max(output_2d[0])-min(output_2d[0])<500):
        output_final = output_2d
        break
    if(210<max(output_2d_2[0])-min(output_2d_2[0])<500):
        output_final = output_2d_2
        break
    if(x>=5):
        print("still need to adjust")
print(output_final)
# pos_2d_pixel_space_demo 即为demo图像经过pca生成的图像，该图像已转换为虚拟相机中的数据格式

fig = plt.figure(figsize=(16, 8))

ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax2d = fig.add_subplot(1, 2, 2)
show3Dpose(demo_rot, ax3d, gt=True)


import matplotlib.pyplot as plt
import numpy as np

vals = output_final
I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

# 绘制关节点


# 绘制关节点之间的连线
if(output_final[0][0]>output_final[8][0]):
    ax2d.scatter(vals[:, 1], -vals[:, 0], s=50, c='black')
    for i in np.arange(len(I)):
        if LR[i]:
            ax2d.plot([vals[I[i], 1], vals[J[i], 1]], [-vals[I[i], 0], -vals[J[i], 0]], lw=2, c='blue')
        else:
            ax2d.plot([vals[I[i], 1], vals[J[i], 1]], [-vals[I[i], 0], -vals[J[i], 0]], lw=2, c='red')
# else:
#     ax2d.scatter(vals[:, 1], vals[:, 0], s=50, c='black')
#     for i in np.arange(len(I)):
#         if LR[i]:
#             ax2d.plot([vals[I[i], 1], vals[J[i], 1]], [vals[I[i], 0], vals[J[i], 0]], lw=2, c='blue')
#         else:
#             ax2d.plot([vals[I[i], 1], vals[J[i], 1]], [vals[I[i], 0], vals[J[i], 0]], lw=2, c='red')

# 设置坐标轴范围和纵横比

# 显示图像
ax2d.plot(vals)
ax2d.axis('auto')
fig.subplots_adjust(wspace=0.2)
plt.show()



