# find all configs in configs/
# config_file=configs/charades.yaml
config_file=configs/charades.yaml

# the dir of the saved weight
# weight_dir=checkpoints/charades/
weight_dir=checkpoints/charades/

# select weight to evaluate
# weight_file=checkpoints/charades/best_charades.pth
weight_file=outputs/charades/pool_model_10e.pth

# test batch size
batch_size=20
# set your gpu id
gpus=9
# number of gpus
gpun=1
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi task on the same machine
master_addr=127.0.0.2
master_port=28578

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
test_net.py --config-file $config_file --ckpt $weight_file OUTPUT_DIR $weight_dir TEST.BATCH_SIZE $batch_size

