# 数据加载与增强模块

import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
import torch
from transformers import BertTokenizer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import (
    IMAGE_SIZE, COLOR_JITTER_BRIGHTNESS, COLOR_JITTER_CONTRAST, COLOR_JITTER_SATURATION,
    RANDOM_AFFINE_DEGREES, RANDOM_AFFINE_TRANSLATION, TEXT_MODEL
)

class AlgaeMultimodalDataset(Dataset):
    """藻类多模态数据集"""
    def __init__(self, data_dir, split='train', tokenizer=None):
        """
        初始化数据集
        Args:
            data_dir: 数据目录
            split: 数据集划分（train/val/test）
            tokenizer: BERT分词器
        """
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained(TEXT_MODEL)

        # 加载数据列表
        self.data = self._load_data()

        # 数据增强
        self.transform = self._get_transforms()

    def _load_data(self):
        """加载数据列表"""
        data = []

        # 加载文本描述
        text_descriptions_path = os.path.join(self.data_dir, 'text_descriptions.json')
        if os.path.exists(text_descriptions_path):
            with open(text_descriptions_path, 'r', encoding='utf-8') as f:
                text_descriptions = json.load(f)
        else:
            text_descriptions = {}

        # 假设数据目录结构为：data_dir/images/label/image.jpg
        images_dir = os.path.join(self.data_dir, 'images')
        if not os.path.exists(images_dir):
            return data

        for label in os.listdir(images_dir):
            label_dir = os.path.join(images_dir, label)
            if not os.path.isdir(label_dir):
                continue

            # 获取文本描述
            if label in text_descriptions:
                desc = text_descriptions[label]['description']
            else:
                desc = f'这是一种藻类，类别为{label}'

            # 加载图像
            for img_name in os.listdir(label_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(label_dir, img_name)
                    # 为每个类别分配一个唯一的标签ID
                    label_id = list(text_descriptions.keys()).index(label) if label in text_descriptions else 0
                    data.append({
                        'image_path': img_path,
                        'text': desc,
                        'label': label_id
                    })
        return data

    def _get_transforms(self):
        """获取数据增强变换"""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ColorJitter(
                    brightness=COLOR_JITTER_BRIGHTNESS,
                    contrast=COLOR_JITTER_CONTRAST,
                    saturation=COLOR_JITTER_SATURATION
                ),
                transforms.RandomAffine(
                    degrees=RANDOM_AFFINE_DEGREES,
                    translate=RANDOM_AFFINE_TRANSLATION
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个样本
        Returns:
            image_tensor: 图像张量
            input_ids: 文本输入ID
            attention_mask: 注意力掩码
            label_id: 标签ID
        """
        sample = self.data[idx]

        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        image_tensor = self.transform(image)

        # 处理文本
        text_inputs = self.tokenizer(
            sample['text'],
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = text_inputs['input_ids'].squeeze()
        attention_mask = text_inputs['attention_mask'].squeeze()

        # 标签
        label_id = torch.tensor(sample['label'], dtype=torch.long)

        return image_tensor, input_ids, attention_mask, label_id

def get_stratified_kfold(data, n_splits=5):
    """
    基于StratifiedKFold的交叉验证划分
    Args:
        data: 数据集列表
        n_splits: 折数
    Returns:
        折划分的索引
    """
    # 提取标签
    labels = [item['label'] for item in data]

    # 初始化StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 生成划分
    folds = []
    for train_idx, val_idx in skf.split(range(len(data)), labels):
        folds.append((train_idx, val_idx))

    return folds

def get_data_loaders(train_dir, val_dir, batch_size=8):
    """
    获取数据加载器
    Args:
        train_dir: 训练数据目录
        val_dir: 验证数据目录
        batch_size: 批次大小
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 创建数据集
    train_dataset = AlgaeMultimodalDataset(train_dir, split='train')
    val_dataset = AlgaeMultimodalDataset(val_dir, split='val')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader
