from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import json
import random
import trimesh
from plyfile import PlyData, PlyElement

def project_pc_to_image(points, resolution=64):
    """project point clouds into 2D image
    :param points: (n, 3) range(-1, 1)
    :return: binary image
    """
    img = []
    for i in range(3):
        canvas = np.zeros((resolution, resolution))
        axis = [0, 1, 2]
        axis.remove(i)
        proj_points = (points[:, axis] + 1) / 2 * resolution
        proj_points = proj_points.astype(np.int)
        canvas[proj_points[:, 0], proj_points[:, 1]] = 1
        img.append(canvas)
    img = np.concatenate(img, axis=1)
    return img


def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)


def rotate_point_cloud(points, transformation_mat):

    new_points = np.dot(transformation_mat, points.T).T

    return new_points


def rotate_point_cloud_by_axis_angle(points, axis, angle_deg):
    """ align 3depn shapes to shapenet coordinates"""
    # angle = math.radians(angle_deg)
    # rot_m = pymesh.Quaternion.fromAxisAngle(axis, angle)
    # rot_m = rot_m.to_matrix()
    rot_m = np.array([[ 2.22044605e-16,  0.00000000e+00,  1.00000000e+00],
                      [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
                      [-1.00000000e+00,  0.00000000e+00,  2.22044605e-16]])

    new_points = rotate_point_cloud(points, rot_m)

    return new_points


def downsample_point_cloud(points, n_pts):
    """downsample points by random choice
    :param points: (n, 3)
    :param n_pts: int
    :return:
    """
    p_idx = random.choices(list(range(points.shape[0])), k=n_pts)
    return points[p_idx]


def upsample_point_cloud(points, n_pts):
    """upsample points by random choice
    :param points: (n, 3)
    :param n_pts: int, > n
    :return:
    """
    p_idx = random.choices(list(range(points.shape[0])), k=n_pts - points.shape[0])
    dup_points = points[p_idx]
    points = np.concatenate([points, dup_points], axis=0)
    return points


def sample_point_cloud_by_n(points, n_pts):
    """resample point cloud to given number of points"""
    if n_pts > points.shape[0]:
        return upsample_point_cloud(points, n_pts)
    elif n_pts < points.shape[0]:
        return downsample_point_cloud(points, n_pts)
    else:
        return points



def collect_data_id(split_dir, classname, phase):
    filename = os.path.join(split_dir, "{}.{}.json".format(classname, phase))
    if not os.path.exists(filename):
        raise ValueError("Invalid filepath: {}".format(filename))

    all_ids = []
    with open(filename, 'r') as fp:
        info = json.load(fp)
    for item in info:
        all_ids.append(item["anno_id"])

    return all_ids



class GANdatasetPartNet(Dataset):
    def __init__(self, phase, data_root, category, n_pts):
        super(GANdatasetPartNet, self).__init__()
        if phase == "validation":
            phase = "val"

        self.phase = phase
        self.aug = phase == "train"

        self.data_root = data_root

        shape_names = collect_data_id(os.path.join(self.data_root, 'partnet_labels/partnet_train_val_test_split'), category, phase)
        self.shape_names = []
        for name in shape_names:
            path = os.path.join(self.data_root, 'partnet_labels/partnet_pc_label', name)
            if os.path.exists(path):
                self.shape_names.append(name)

        self.n_pts = n_pts
        self.raw_n_pts = self.n_pts // 2

        self.rng = random.Random(1234)

    @staticmethod
    def load_point_cloud(path):
        pc = trimesh.load(path)
        pc = pc.vertices / 2.0 # scale to unit sphere
        return pc

    @staticmethod
    def read_point_cloud_part_label(path):
        with open(path, 'r') as fp:
            labels = fp.readlines()
        labels = np.array([int(x) for x in labels])
        return labels

    def random_rm_parts(self, raw_pc, part_labels):
        part_ids = sorted(np.unique(part_labels).tolist())
        if self.phase == "train":
            random.shuffle(part_ids)
            n_part_keep = random.randint(1, max(1, len(part_ids) - 1))
        else:
            self.rng.shuffle(part_ids)
            n_part_keep = self.rng.randint(1, max(1, len(part_ids) - 1))
        part_ids_keep = part_ids[:n_part_keep]
        point_idx = []
        for i in part_ids_keep:
            point_idx.extend(np.where(part_labels == i)[0].tolist())
        raw_pc = raw_pc[point_idx]
        return raw_pc, n_part_keep

    def __getitem__(self, index):
        raw_shape_name = self.shape_names[index]
        raw_ply_path = os.path.join(self.data_root, 'partnet_data', raw_shape_name, 'point_sample/ply-10000.ply')
        raw_pc = self.load_point_cloud(raw_ply_path)

        raw_label_path = os.path.join(self.data_root, 'partnet_labels/partnet_pc_label', raw_shape_name, 'label-merge-level1-10000.txt')
        part_labels = self.read_point_cloud_part_label(raw_label_path)
        raw_pc, n_part_keep = self.random_rm_parts(raw_pc, part_labels)
        raw_pc = sample_point_cloud_by_n(raw_pc, self.raw_n_pts)
        raw_pc = torch.tensor(raw_pc, dtype=torch.float32).transpose(1, 0)

        real_shape_name = self.shape_names[index]
        real_ply_path = os.path.join(self.data_root, 'partnet_data', real_shape_name, 'point_sample/ply-10000.ply')
        real_pc = self.load_point_cloud(real_ply_path)
        real_pc = sample_point_cloud_by_n(real_pc, self.n_pts)
        real_pc = torch.tensor(real_pc, dtype=torch.float32).transpose(1, 0)

        return {"raw": raw_pc, "real": real_pc, "raw_id": raw_shape_name, "real_id": real_shape_name,
                'n_part_keep': n_part_keep, 'idx': index}

    def __len__(self):
        return len(self.shape_names)



