
python train.py --distribution_type 'multi' \
    --dataroot /path/to/ShapeNetCore.v2.PC15k/ \
    --category chair \
    --experiment_name /path/to/experiments \
    --model_type 'DiT-L/4' \
    --window_size 4 --window_block_indexes '0,3,6,9,12,15,18' \
    --bs 16 \
    --voxel_size 32 \
    --lr 1e-4 \
    --use_tb
