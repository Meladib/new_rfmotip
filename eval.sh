CUDA_VISIBLE_DEVICES=2 python submit_and_evaluate.py \
  --config configs/rf_detrV5_motip_dancetrack.yaml \
  --inference-model rfdetr_dancetrack_motip/checkpoint_7.pth \
  --outputs-dir outputsV5/rfmotip_dancetrack_V5/

