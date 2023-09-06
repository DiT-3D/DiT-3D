import warnings
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import open3d as o3d
import os
import numpy as np

import hashlib
import torch
import matplotlib.pyplot as plt

synset_to_label = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}

# Label to Synset mapping (for ShapeNet core classes)
label_to_synset = {v: k for k, v in synset_to_label.items()}

def _convert_categories(categories):
    assert categories is not None, 'List of categories cannot be empty!'
    if not (c in synset_to_label.keys() + label_to_synset.keys()
            for c in categories):
        warnings.warn('Some or all of the categories requested are not part of \
            ShapeNetCore. Data loading may fail if these categories are not avaliable.')
    synsets = [label_to_synset[c] if c in label_to_synset.keys()
               else c for c in categories]
    return synsets


class ShapeNet_Multiview_Points(Dataset):
    def __init__(self, root_pc:str, root_views: str, cache: str, categories: list = ['chair'], split: str= 'val',
                 npoints=2048, sv_samples=800, all_points_mean=None, all_points_std=None, get_image=False):
        self.root = Path(root_views)
        self.split = split
        self.get_image = get_image
        params = {
            'cat': categories,
            'npoints': npoints,
            'sv_samples': sv_samples,
        }
        params = tuple(sorted(pair for pair in params.items()))
        self.cache_dir = Path(cache) / 'svpoints/{}/{}'.format('_'.join(categories), hashlib.md5(bytes(repr(params), 'utf-8')).hexdigest())

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths = []
        self.synset_idxs = []
        self.synsets = _convert_categories(categories)
        self.labels = [synset_to_label[s] for s in self.synsets]
        self.npoints = npoints
        self.sv_samples = sv_samples

        self.all_points = []
        self.all_points_sv = []

        # loops through desired classes
        for i in range(len(self.synsets)):

            syn = self.synsets[i]
            class_target = self.root / syn
            if not class_target.exists():
                raise ValueError('Class {0} ({1}) was not found at location {2}.'.format(
                    syn, self.labels[i], str(class_target)))


            sub_path_pc = os.path.join(root_pc, syn, split)
            if not os.path.isdir(sub_path_pc):
                print("Directory missing : %s" % sub_path_pc)
                continue

            self.all_mids = []
            self.imgs = []
            for x in os.listdir(sub_path_pc):
                if not x.endswith('.npy'):
                    continue
                self.all_mids.append(os.path.join(split, x[:-len('.npy')]))

            for mid in tqdm(self.all_mids):
                # obj_fname = os.path.join(sub_path, x)
                obj_fname = os.path.join(root_pc, syn, mid + ".npy")
                cams_pths = list((self.root/ syn/ mid.split('/')[-1]).glob('*_cam_params.npz'))
                if len(cams_pths) < 20:
                    continue
                point_cloud = np.load(obj_fname)
                sv_points_group = []
                img_path_group = []
                (self.cache_dir / (mid.split('/')[-1])).mkdir(parents=True, exist_ok=True)
                success = True
                for i, cp in enumerate(cams_pths):
                    cp = str(cp)
                    vp = cp.split('cam_params')[0] + 'depth.png'
                    depth_minmax_pth = cp.split('_cam_params')[0] + '.npy'
                    cache_pth = str(self.cache_dir / mid.split('/')[-1] / os.path.basename(depth_minmax_pth) )

                    cam_params = np.load(cp)
                    extr = cam_params['extr']
                    intr = cam_params['intr']

                    self.transform = DepthToSingleViewPoints(cam_ext=extr, cam_int=intr)

                    try:
                        sv_point_cloud = self._render(cache_pth, vp, depth_minmax_pth)

                        img_path_group.append(vp)

                        sv_points_group.append(sv_point_cloud)
                    except Exception as e:
                        print(e)
                        success=False
                        break
                if not success:
                    continue
                self.all_points_sv.append(np.stack(sv_points_group, axis=0))
                self.all_points.append(point_cloud)
                self.imgs.append(img_path_group)

        self.all_points = np.stack(self.all_points, axis=0)

        self.all_points_sv = np.stack(self.all_points_sv, axis=0)
        if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        else:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)
            self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:,:10000]
        self.test_points = self.all_points[:,10000:]
        self.all_points_sv = (self.all_points_sv - self.all_points_mean) / self.all_points_std

    def get_pc_stats(self, idx):

        return self.all_points_mean.reshape(1,1, -1), self.all_points_std.reshape(1,1, -1)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.all_points)

    def __getitem__(self, index):


        tr_out = self.train_points[index]
        tr_idxs = np.random.choice(tr_out.shape[0], self.npoints)
        tr_out = tr_out[tr_idxs, :]

        gt_points = self.test_points[index][:self.npoints]

        m, s = self.get_pc_stats(index)

        sv_points = self.all_points_sv[index]

        idxs = np.arange(0, sv_points.shape[-2])[:self.sv_samples]#np.random.choice(sv_points.shape[0], 500, replace=False)

        data = torch.cat([torch.from_numpy(sv_points[:,idxs]).float(),
                          torch.zeros(sv_points.shape[0], self.npoints - idxs.shape[0], sv_points.shape[2])], dim=1)
        masks = torch.zeros_like(data)
        masks[:,:idxs.shape[0]] = 1

        res = {'train_points': torch.from_numpy(tr_out).float(),
                'test_points': torch.from_numpy(gt_points).float(),
                'sv_points': data,
                'masks': masks,
                'std': s, 'mean': m,
                'idx': index,
               'name':self.all_mids[index]
                }

        if self.split != 'train' and self.get_image:

            img_lst = []
            for n in range(self.all_points_sv.shape[1]):

                img = torch.from_numpy(plt.imread(self.imgs[index][n])).float().permute(2,0,1)[:3]

                img_lst.append(img)

            img = torch.stack(img_lst, dim=0)

            res['image'] = img

        return res



    def _render(self, cache_path, depth_pth, depth_minmax_pth):
        # if not os.path.exists(cache_path.split('.npy')[0] + '_color.png') and os.path.exists(cache_path):
        #
        #     os.remove(cache_path)

        if os.path.exists(cache_path):
            data = np.load(cache_path)
        else:

            data, depth = self.transform(depth_pth, depth_minmax_pth)
            assert data.shape[0] > 600, 'Only {} points found'.format(data.shape[0])
            data = data[np.random.choice(data.shape[0], 600, replace=False)]
            np.save(cache_path, data)

        return data




class DepthToSingleViewPoints(object):
    '''
    render a view then save mask
    '''
    def __init__(self, cam_ext, cam_int):

        self.cam_ext = cam_ext.reshape(4,4)
        self.cam_int = cam_int.reshape(3,3)


    def __call__(self, depth_pth, depth_minmax_pth):

        depth_minmax = np.load(depth_minmax_pth)
        depth_img = plt.imread(depth_pth)[...,0]
        mask = np.where(depth_img == 0, -1.0, 1.0)
        depth_img = 1 - depth_img
        depth_img = (depth_img * (np.max(depth_minmax) - np.min(depth_minmax)) + np.min(depth_minmax)) * mask

        intr = o3d.camera.PinholeCameraIntrinsic(depth_img.shape[0], depth_img.shape[1],
                                                 self.cam_int[0, 0], self.cam_int[1, 1], self.cam_int[0,2],
                                                 self.cam_int[1,2])

        depth_im = o3d.geometry.Image(depth_img.astype(np.float32, copy=False))

        # rgbd_im = o3d.geometry.RGBDImage.create_from_color_and_depth(color_im, depth_im)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_im, intr, self.cam_ext, depth_scale=1.)
        pc =  np.asarray(pcd.points)

        return pc, depth_img

    def __repr__(self):
        return 'MeshToMaskedVoxel_'+str(self.radius)+str(self.resolution)+str(self.elev )+str(self.azim)+str(self.img_size )

