CUDA_VISIBLE_DEVICES=4,5,6,7 \
accelerate launch --num_processes=4 train.py \
    --data-root /data/pos+mot/Datadir/ \
    --exp-name rfmotip_dancetrack_ctsv \
    --config-path ./configs/rf_detrV5_motip_dancetrack.yaml

