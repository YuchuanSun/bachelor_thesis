# 可视化分析模块
# 用于生成论文所需的各种可视化图表

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import umap
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io
import base64
from src.visualization import VisualizationTool

# 创建可视化工具实例
viz_tool = VisualizationTool()

# 1. 生成模型架构图
viz_tool.plot_model_architecture()

# 2. 绘制训练指标曲线
train_losses = [2.5, 2.0, 1.5, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25]
val_losses = [2.4, 1.9, 1.6, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35]
train_accs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9]
val_accs = [0.25, 0.35, 0.45, 0.55, 0.65, 0.7, 0.75, 0.8, 0.83, 0.85]
viz_tool.plot_training_metrics(train_losses, val_losses, train_accs, val_accs)

# 3. 绘制数据分布图
class_counts = {i: 100 for i in range(16)}  # 示例数据
viz_tool.plot_data_distribution(class_counts)
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from configs.default_config import ALGAE_DESCRIPTIONS, RESULT_DIR

class VisualizationTool:
    """可视化工具类"""
    
    def __init__(self):
        """初始化可视化工具"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        # 创建结果目录
        os.makedirs(RESULT_DIR, exist_ok=True)
    
    def plot_training_metrics(self, train_losses, val_losses, train_accuracies=None, val_accuracies=None, save_path=None):
        """
        绘制训练指标曲线
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            train_accuracies: 训练准确率列表
            val_accuracies: 验证准确率列表
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.title('损失曲线')
        plt.xlabel('轮次')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制准确率曲线
        if train_accuracies and val_accuracies:
            plt.subplot(1, 2, 2)
            plt.plot(train_accuracies, label='训练准确率')
            plt.plot(val_accuracies, label='验证准确率')
            plt.title('准确率曲线')
            plt.xlabel('轮次')
            plt.ylabel('准确率')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(RESULT_DIR, 'training_metrics.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_embedding_space(self, embeddings, labels, method='tsne', save_path=None):
        """
        绘制嵌入空间可视化
        Args:
            embeddings: 嵌入向量，形状为[num_samples, embedding_dim]
            labels: 标签列表
            method: 降维方法，可选'tsne'或'umap'
            save_path: 保存路径
        """
        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embedded = reducer.fit_transform(embeddings)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedded = reducer.fit_transform(embeddings)
        else:
            reducer = PCA(n_components=2)
            embedded = reducer.fit_transform(embeddings)
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        # 绘制散点图
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            embedded[:, 0], 
            embedded[:, 1], 
            c=labels, 
            cmap=plt.cm.get_cmap('tab20', num_classes),
            s=50, 
            alpha=0.7
        )
        
        # 添加颜色条和图例
        cbar = plt.colorbar(scatter, ticks=unique_labels)
        cbar.set_label('藻类类别')
        
        # 添加类别名称
        class_names = [list(ALGAE_DESCRIPTIONS.keys())[int(label)] for label in unique_labels]
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i/num_classes), markersize=10) for i in range(num_classes)]
        plt.legend(handles, class_names, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(f'嵌入空间可视化 ({method.upper()})')
        plt.xlabel('维度1')
        plt.ylabel('维度2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(RESULT_DIR, f'embedding_space_{method}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        绘制混淆矩阵
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 获取类别名称
        class_names = list(ALGAE_DESCRIPTIONS.keys())
        
        # 绘制混淆矩阵
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(RESULT_DIR, 'confusion_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_model_architecture(self, save_path=None):
        """
        绘制模型架构图
        Args:
            save_path: 保存路径
        """
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        nodes = [
            '输入图像', 'ConvNeXt-Tiny', '视觉投影层', '图像嵌入',
            '输入文本', 'BERT-base-chinese', '文本投影层', '文本嵌入',
            '对比学习', '温度参数'
        ]
        
        for node in nodes:
            G.add_node(node)
        
        # 添加边
        edges = [
            ('输入图像', 'ConvNeXt-Tiny'),
            ('ConvNeXt-Tiny', '视觉投影层'),
            ('视觉投影层', '图像嵌入'),
            ('输入文本', 'BERT-base-chinese'),
            ('BERT-base-chinese', '文本投影层'),
            ('文本投影层', '文本嵌入'),
            ('图像嵌入', '对比学习'),
            ('文本嵌入', '对比学习'),
            ('温度参数', '对比学习')
        ]
        
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        
        # 绘制图形
        plt.figure(figsize=(14, 8))
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # 绘制节点和边
        nx.draw(G, pos, with_labels=True, node_size=3000, 
                node_color='lightblue', font_size=10, font_weight='bold',
                arrowsize=20, edge_color='gray')
        
        plt.title('多模态双塔Siamese网络架构')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(RESULT_DIR, 'model_architecture.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_sample_predictions(self, images, true_labels, pred_labels, probabilities, save_path=None):
        """
        绘制样本预测结果
        Args:
            images: 图像列表
            true_labels: 真实标签
            pred_labels: 预测标签
            probabilities: 预测概率
            save_path: 保存路径
        """
        num_samples = min(8, len(images))
        rows = (num_samples + 3) // 4
        cols = min(4, num_samples)
        
        plt.figure(figsize=(16, 4 * rows))
        
        for i in range(num_samples):
            plt.subplot(rows, cols, i + 1)
            
            # 显示图像
            if isinstance(images[i], torch.Tensor):
                img = images[i].permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())
                plt.imshow(img)
            else:
                plt.imshow(images[i])
            
            # 获取类别名称
            true_name = ALGAE_DESCRIPTIONS.get(true_labels[i], str(true_labels[i]))
            pred_name = ALGAE_DESCRIPTIONS.get(pred_labels[i], str(pred_labels[i]))
            
            # 标题
            title = f'真实: {true_name}\n预测: {pred_name}\n概率: {probabilities[i]:.2f}'
            plt.title(title, fontsize=10)
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(RESULT_DIR, 'sample_predictions.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_data_distribution(self, class_counts, save_path=None):
        """
        绘制数据分布
        Args:
            class_counts: 各类别样本数量
            save_path: 保存路径
        """
        class_names = list(ALGAE_DESCRIPTIONS.keys())
        counts = [class_counts.get(c, 0) for c in class_names]
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(class_names)), counts)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        plt.title('数据集类别分布')
        plt.xlabel('藻类类别')
        plt.ylabel('样本数量')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(RESULT_DIR, 'data_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_attention_heatmap(self, attention_weights, tokens, save_path=None):
        """
        绘制注意力热力图
        Args:
            attention_weights: 注意力权重
            tokens: 分词结果
            save_path: 保存路径
        """
        # 获取第一层第一个注意力头的权重
        attention = attention_weights[0, 0].cpu().detach().numpy()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(attention, cmap='viridis', xticklabels=tokens, yticklabels=tokens)
        
        plt.title('BERT注意力热力图')
        plt.xlabel('Token位置')
        plt.ylabel('Token位置')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(RESULT_DIR, 'attention_heatmap.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def generate_all_visualizations(self, model, dataloader, train_metrics=None, val_metrics=None):
        """
        生成所有可视化图表
        Args:
            model: 模型
            dataloader: 数据加载器
            train_metrics: 训练指标
            val_metrics: 验证指标
        """
        visualizations = {}
        
        # 1. 模型架构图
        arch_path = self.plot_model_architecture()
        visualizations['model_architecture'] = arch_path
        print(f'生成模型架构图: {arch_path}')
        
        # 2. 训练指标曲线
        if train_metrics and val_metrics:
            train_losses = train_metrics.get('loss', [])
            val_losses = val_metrics.get('loss', [])
            train_accs = train_metrics.get('accuracy', [])
            val_accs = val_metrics.get('accuracy', [])
            
            metrics_path = self.plot_training_metrics(
                train_losses, val_losses, train_accs, val_accs
            )
            visualizations['training_metrics'] = metrics_path
            print(f'生成训练指标曲线: {metrics_path}')
        
        # 3. 数据分布
        class_counts = {}
        for batch in dataloader:
            labels = batch.get('label', [])
            for label in labels:
                label = label.item() if isinstance(label, torch.Tensor) else label
                class_counts[label] = class_counts.get(label, 0) + 1
        
        dist_path = self.plot_data_distribution(class_counts)
        visualizations['data_distribution'] = dist_path
        print(f'生成数据分布图: {dist_path}')
        
        # 4. 嵌入空间可视化
        # 收集嵌入向量和标签
        embeddings = []
        labels = []
        
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch.get('image')
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
                batch_labels = batch.get('label')
                
                if images is not None:
                    img_embeds, _ = model(image=images)
                    embeddings.append(img_embeds.cpu().numpy())
                    labels.extend(batch_labels.cpu().numpy())
        
        if embeddings:
            embeddings = np.vstack(embeddings)
            tsne_path = self.plot_embedding_space(embeddings, labels, method='tsne')
            umap_path = self.plot_embedding_space(embeddings, labels, method='umap')
            visualizations['embedding_tsne'] = tsne_path
            visualizations['embedding_umap'] = umap_path
            print(f'生成嵌入空间可视化: {tsne_path}, {umap_path}')
        
        return visualizations

# 示例用法
if __name__ == '__main__':
    viz_tool = VisualizationTool()
    
    # 示例：生成模型架构图
    viz_tool.plot_model_architecture()
    
    # 示例：模拟训练指标
    train_losses = [2.5, 2.0, 1.5, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25]
    val_losses = [2.4, 1.9, 1.6, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35]
    train_accs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9]
    val_accs = [0.25, 0.35, 0.45, 0.55, 0.65, 0.7, 0.75, 0.8, 0.83, 0.85]
    
    viz_tool.plot_training_metrics(train_losses, val_losses, train_accs, val_accs)
    
    # 示例：模拟数据分布
    class_counts = {i: np.random.randint(50, 200) for i in range(16)}
    viz_tool.plot_data_distribution(class_counts)
    
    print('可视化生成完成！')
