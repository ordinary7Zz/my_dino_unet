python gradcam_single_image_seg.py \
    --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/baseline/dino_unet_train_dataset_4_epoch_50.pth \
    --image_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/train/images/ThyroidXL_train_00002730_C9690598_1.png \
    --mask_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/train/masks/ThyroidXL_train_00002730_C9690598_1.png \
    --output_dir ./pyradiomics_dice/gradcam/BM \
    --img_size 224 \
    --dino_pretrained True \
    --alpha 0.3 \
    --target_layer up1 \
    --smooth_sigma_ratio 0.1 \
    --gamma 0.4 \
    --saturation_scale 1.5 \
    --output_type all

python gradcam_single_image_seg.py \
    --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/baseline/dino_unet_train_dataset_4_epoch_50.pth \
    --image_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images/TN3K_test_0040.jpg \
    --mask_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/masks/TN3K_test_0040.jpg \
    --output_dir ./pyradiomics_dice/gradcam/BM \
    --img_size 224 \
    --dino_pretrained True \
    --alpha 0.3 \
    --target_layer up1 \
    --smooth_sigma_ratio 0.1 \
    --gamma 0.4 \
    --saturation_scale 1.5 \
    --output_type all

python gradcam_single_image_seg.py \
    --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/baseline/dino_unet_train_dataset_4_epoch_50.pth \
    --image_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/images/TN5K_test_003323.jpg \
    --mask_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/masks/TN5K_test_003323.png \
    --output_dir ./pyradiomics_dice/gradcam/BM \
    --img_size 224 \
    --dino_pretrained True \
    --alpha 0.3 \
    --target_layer up1 \
    --smooth_sigma_ratio 0.1 \
    --gamma 0.4 \
    --saturation_scale 1.5 \
    --output_type all

python gradcam_single_image_seg.py \
    --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/checkpoints/baseline/dino_unet_train_dataset_4_epoch_50.pth \
    --image_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images/ThyroidXL_test_00001873_6923593C_2.png \
    --mask_path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/masks/ThyroidXL_test_00001873_6923593C_2.png \
    --output_dir ./pyradiomics_dice/gradcam/BM \
    --img_size 224 \
    --dino_pretrained True \
    --alpha 0.3 \
    --target_layer up1 \
    --smooth_sigma_ratio 0.1 \
    --gamma 0.4 \
    --saturation_scale 1.5 \
    --output_type all
