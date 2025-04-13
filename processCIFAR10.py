import os
import numpy as np
import pickle
import urllib.request
from PIL import Image

# CIFAR-10 数据集元信息
CIFAR10_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CIFAR10_FILES = [
    ("cifar-10-python.tar.gz", "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
]

def download_and_extract_cifar10(save_dir):
    """下载并解压 CIFAR-10 数据集"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 下载压缩文件
    file_name, url = CIFAR10_FILES[0]
    file_path = os.path.join(save_dir, file_name)
    
    if not os.path.exists(file_path):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete!")
    
    # 解压文件
    print("Extracting files...")
    import tarfile
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=save_dir)
    print("Extraction complete!")

def load_batch(file_path):
    """加载 CIFAR-10 的批量数据"""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch

def save_images_from_batch(batch, output_dir, split='train'):
    """保存单批次图片到文件夹"""
    images = batch[b'data']
    labels = batch[b'labels']
    filenames = batch[b'filenames']
    
    # CIFAR-10 图片格式转换
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC格式
    
    for idx, (image, label, filename) in enumerate(zip(images, labels, filenames)):
        # 创建类别文件夹路径
        class_name = CIFAR10_LABELS[label]
        class_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # 转换并保存图片
        img = Image.fromarray(image)
        img_path = os.path.join(class_dir, filename.decode('utf-8'))
        img.save(img_path)

def organize_cifar10(raw_dir, output_dir):
    """组织数据集到分类文件夹"""
    # 处理训练集 (共5个批次)
    for i in range(1, 6):
        batch_path = os.path.join(raw_dir, f'cifar-10-batches-py', f'data_batch_{i}')
        print(f"Processing {batch_path}")
        batch = load_batch(batch_path)
        save_images_from_batch(batch, output_dir, split='train')
    
    # 处理测试集
    test_batch_path = os.path.join(raw_dir, 'cifar-10-batches-py', 'test_batch')
    print(f"Processing {test_batch_path}")
    test_batch = load_batch(test_batch_path)
    save_images_from_batch(test_batch, output_dir, split='test')

if __name__ == "__main__":
    # 参数设置
    RAW_DATA_DIR = "./data/cifar10_raw"  # 原始文件存储路径
    FINAL_DATA_DIR = "./data/cifar10"    # 整理后的最终路径
    
    # 执行下载和解压
    download_and_extract_cifar10(RAW_DATA_DIR)
    
    # 组织数据到分类文件夹
    organize_cifar10(RAW_DATA_DIR, FINAL_DATA_DIR)
    
    print("CIFAR-10 数据集整理完成！")
    print(f"整理后路径结构：\n{FINAL_DATA_DIR}")
    print("   ├── train/")
    print("   │   ├── airplane/")
    print("   │   ├── automobile/")
    print("   │   └── ...")
    print("   └── test/")
    print("       ├── airplane/")
    print("       └── ...")