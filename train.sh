python setup.py develop --no_cuda_ext
#python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro.yml --launcher pytorch
# 这是正确的命令（用于单GPU）
#python basicsr/train.py -opt /home/zqk/EVSSM-ori/options/train/GoPro.yml
torchrun --nproc_per_node=2 --master_port=29500 basicsr/train.py -opt /home/zqk/EVSSM-ori/options/train/GoPro.yml --launcher pytorch