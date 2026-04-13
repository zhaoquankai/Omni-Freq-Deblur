

echo "---  ---"



torchrun --nproc_per_node=2 --master_port=29500 test.py \
    --test_model "Omni_freq_deblur_arch/GSblur.pth" \
    --data_dir "GOPRO/test" \
    --width 48 \
    --num_workers 8