python shap_single_image_seg.py \
  --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/baseline/dino_unet_train_dataset_4_epoch_50.pth \
  --image_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/FangDai/PTC/A_b17171017081122.png \
  --mask_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions/FangDai/PTC/A_b17171017081122.png \
  --output_dir ./pyradiomics_dice/shap_single_out/FTCPTC \
  --img_size 224 \
  --dino_pretrained True \
  --background_mode zeros \
  --focus_percentile 10 \
  --overlay_alpha_scale 0.95
