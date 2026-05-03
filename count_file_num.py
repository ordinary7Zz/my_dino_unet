import os

def count_files_in_directories(directories):
    total_files = 0
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            total_files += len(files)
    return total_files

# 示例使用
directories = [
    '/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_5/train/images',
]

file_count = count_files_in_directories(directories)
print(f"Total number of files: {file_count}")
