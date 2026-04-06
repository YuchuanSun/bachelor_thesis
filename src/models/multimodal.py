# 双塔主模型（ConvNeXt + BERT）

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import BertModel

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from configs.default_config import (
    VISION_MODEL, VISION_EMBED_DIM, TEXT_MODEL, TEXT_EMBED_DIM, FREEZE_BERT_LAYERS,
    TEMPERATURE
)

class MultimodalModel(nn.Module):
    """多模态双塔模型"""
    def __init__(self):
        """初始化多模态模型"""
        super(MultimodalModel, self).__init__()
        # 视觉塔（Image Encoder）
        self.vision_encoder = timm.create_model(VISION_MODEL, pretrained=True, num_classes=0)
        # 获取视觉编码器的输出维度
        vision_output_dim = self.vision_encoder.num_features
        # 投影层：将视觉特征投影到768维
        self.vision_proj = nn.Linear(vision_output_dim, VISION_EMBED_DIM)

        # 文本塔（Text Encoder）
        self.text_encoder = BertModel.from_pretrained(TEXT_MODEL)
        # 冻结BERT前10层参数
        for i, layer in enumerate(self.text_encoder.encoder.layer):
            if i < FREEZE_BERT_LAYERS:
                for param in layer.parameters():
                    param.requires_grad = False
        # 投影层：将文本特征投影到768维
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, TEXT_EMBED_DIM)

        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / TEMPERATURE))

    def forward(self, image=None, input_ids=None, attention_mask=None):
        """
        前向传播
        Args:
            image: 图像张量，形状为[batch_size, 3, 224, 224]
            input_ids: 文本输入ID，形状为[batch_size, max_length]
            attention_mask: 注意力掩码，形状为[batch_size, max_length]
        Returns:
            image_embeds: 图像嵌入，形状为[batch_size, 768]
            text_embeds: 文本嵌入，形状为[batch_size, 768]
        """
        image_embeds = None
        text_embeds = None

        # 处理图像
        if image is not None:
            # 视觉编码器前向传播
            # image: [batch_size, 3, 224, 224] -> vision_features: [batch_size, vision_output_dim]
            vision_features = self.vision_encoder(image)
            # 投影到768维
            # vision_features: [batch_size, vision_output_dim] -> image_embeds: [batch_size, 768]
            image_embeds = self.vision_proj(vision_features)
            # L2归一化
            # image_embeds: [batch_size, 768] -> [batch_size, 768]
            image_embeds = F.normalize(image_embeds, p=2, dim=1)

        # 处理文本
        if input_ids is not None and attention_mask is not None:
            # 文本编码器前向传播
            # input_ids: [batch_size, max_length], attention_mask: [batch_size, max_length]
            # -> text_outputs.last_hidden_state: [batch_size, max_length, hidden_size]
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # 提取[CLS]向量
            # text_outputs.last_hidden_state: [batch_size, max_length, hidden_size] -> cls_token: [batch_size, hidden_size]
            cls_token = text_outputs.last_hidden_state[:, 0, :]
            # 投影到768维
            # cls_token: [batch_size, hidden_size] -> text_embeds: [batch_size, 768]
            text_embeds = self.text_proj(cls_token)
            # L2归一化
            # text_embeds: [batch_size, 768] -> [batch_size, 768]
            text_embeds = F.normalize(text_embeds, p=2, dim=1)

        return image_embeds, text_embeds

    def get_logit_scale(self):
        """获取温度参数的指数值"""
        return self.logit_scale.exp()
