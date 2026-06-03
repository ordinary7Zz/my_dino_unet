python shap_single_image_seg.py \
  --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/baseline/dino_unet_train_dataset_4_epoch_50.pth \
  --image_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/ \
  --mask_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions/ \
  --output_dir ./pyradiomics_dice/shap_single_out/FTCPTC \
  --img_size 224 \
  --dino_pretrained True \
  --background_mode zeros \
  --focus_percentile 80 \
  --overlay_alpha_scale 0.9
