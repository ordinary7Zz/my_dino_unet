python shap_single_image_seg.py \
  --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/baseline/dino_unet_train_dataset_4_epoch_50.pth \
  --image_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images/ThyroidXL_test_00000215_564DFC1D_0.png \
  --mask_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/masks/ThyroidXL_test_00000215_564DFC1D_0.png \
  --output_dir ./pyradiomics_dice/shap_single_out \
  --img_size 224 \
  --dino_pretrained True \
  --background_mode zeros \
  --focus_percentile 80 \
  --overlay_alpha_scale 0.9
