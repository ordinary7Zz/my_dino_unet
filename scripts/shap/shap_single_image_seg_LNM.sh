python shap_single_image_seg.py \
  --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/train_Lymph_Node_Metastasis/dino_unet_Lymph_Node_Metastasis_epoch_40.pth \
  --image_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/images/22_Benign_center1.png \
  --mask_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/masks/22_Benign_center1.png \
  --output_dir ./pyradiomics_dice/shap_single_out/LNM_CN01 \
  --img_size 224 \
  --dino_pretrained True \
  --background_mode zeros \
  --focus_percentile 80 \
  --overlay_alpha_scale 0.9
