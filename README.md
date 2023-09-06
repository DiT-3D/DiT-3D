# DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation

### [**Project Page**](https://dit-3d.github.io/) | [**Paper**](https://arxiv.org/abs/2307.01831) | [**Hugging Face**](https://huggingface.co/papers/2307.01831) | [**Twitter**](https://twitter.com/_akhaliq/status/1676789843689455618)


🔥🔥🔥DiT-3D is a novel Diffusion Transformer for 3D shape generation, which can directly operate the denoising process on voxelized point clouds using plain Transformers.


<div align="center">
  <img width="100%" alt="DiT-3D Illustration" src="assets/framework.png">
</div>


[**DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation**](https://arxiv.org/abs/2307.01831)
<br>Shentong Mo, Enze Xie, Ruihang Chu, Lanqing Hong, Matthias Nießner, Zhenguo Li<br>
arXiv 2023.


## Requirements

Make sure the following environments are installed.

```
python==3.6
pytorch==1.7.1
torchvision==0.8.2
cudatoolkit==11.0
matplotlib==2.2.5
tqdm==4.32.1
open3d==0.9.0
trimesh=3.7.12
scipy==1.5.1
```

Install PyTorchEMD by
```
cd metrics/PyTorchEMD
python setup.py install
cp build/**/emd_cuda.cpython-36m-x86_64-linux-gnu.so .
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Or please simply run

```
pip install -r requirements.txt
```


## Data

For generation, we use ShapeNet point cloud, which can be downloaded [here](https://github.com/stevenygd/PointFlow).


## Pretrained models
Pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1sRnCObeDal3HeD-h_1L28gY9qE6fP0ZZ?usp=sharing).

More pretrained weights will be released soon. Please stay tuned!



## Training

For training the DiT-3D model, please run

```bash
$ python train_generation.py --distribution_type 'multi' \
    --dataroot /path/to/ShapeNetCore.v2.PC15k/ \
    --category chair|car|airplane \
    --experiment_name /path/to/experiments \
    --model_type 'DiT-S/4' \
    --bs 16 \
    --voxel_size 32 \
    --lr 1e-4 \
    --use_tb
# for using window attention, please add flags below
    --window_size 2 --window_block_indexes '0,3,6,9'
```


## Testing

For testing, simply run

```bash
$ python test_generation.py --dataroot /path/to/ShapeNetCore.v2.PC15k/ \
    --category chair|car|airplane \
    --model_type 'DiT-S/4' \
    --voxel_size 32 \
    --model MODEL_PATH
```



## Citation

If you find this repository useful, please cite our paper:
```
@article{mo2023dit3d,
  title = {DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation},
  author = {Shentong Mo and Enze Xie and Ruihang Chu and Lanqing Hong and Matthias Nießner and Zhenguo Li},
  journal = {arXiv preprint arXiv: 2307.01831},
  year = {2023}
}
```

## Acknowledgement


This repo is inspired by [DiT](https://github.com/facebookresearch/DiT) and [PVD](https://github.com/alexzhou907/PVD). Thanks for their wonderful works.




