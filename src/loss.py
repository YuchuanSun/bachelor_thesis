# 损失函数模块

import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """InfoNCE损失函数（对称对比损失）"""
    def __init__(self, temperature=0.07):
        """
        初始化InfoNCE损失
        Args:
            temperature: 温度参数
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        """
        计算InfoNCE损失
        Args:
            image_embeddings: 图像特征，形状为[batch_size, embed_dim]
            text_embeddings: 文本特征，形状为[batch_size, embed_dim]
        Returns:
            对称对比损失值
        """
        # 计算相似度矩阵
        # 图像特征和文本特征已经过L2归一化，所以点积即为余弦相似度
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())

        # 应用温度缩放
        logits = similarity_matrix / self.temperature

        # 创建标签（对角线为正样本）
        batch_size = image_embeddings.size(0)
        labels = torch.arange(batch_size, device=image_embeddings.device)

        # 计算图像到文本的交叉熵损失
        loss_ij = F.cross_entropy(logits, labels)

        # 计算文本到图像的交叉熵损失
        loss_ji = F.cross_entropy(logits.t(), labels)

        # 取均值作为最终损失
        loss = (loss_ij + loss_ji) / 2

        return loss

class CrossEntropyLoss(nn.Module):
    """传统交叉熵损失"""
    def __init__(self):
        """初始化交叉熵损失"""
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        计算交叉熵损失
        Args:
            logits: 模型输出的logits，形状为[batch_size, num_classes]
            labels: 真实标签，形状为[batch_size]
        Returns:
            交叉熵损失值
        """
        return self.loss_fn(logits, labels)
