# 模型评估脚本
# 使用独立测试集评估模型性能，计算真实的性能指标

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.models.multimodal import MultimodalModel
from src.dataset import AlgaeMultimodalDataset
from configs.default_config import DEVICE, BATCH_SIZE, ALGAE_DESCRIPTIONS

def load_dataset_split():
    """加载数据集划分结果"""
    data_dir = os.path.join(project_root, 'data')
    split_file = os.path.join(data_dir, 'dataset_split.json')
    
    if not os.path.exists(split_file):
        print(f'错误：{split_file} 文件不存在，请先运行 dataset_splitter.py')
        return None
    
    with open(split_file, 'r', encoding='utf-8') as f:
        split_info = json.load(f)
    
    print('数据集划分结果加载完成！')
    print(f'训练集: {len(split_info["train"])} 张图像')
    print(f'验证集: {len(split_info["val"])} 张图像')
    print(f'测试集: {len(split_info["test"])} 张图像')
    
    return split_info

def create_test_dataset(split_info):
    """创建测试数据集"""
    data_dir = os.path.join(project_root, 'data')
    images_dir = os.path.join(data_dir, 'images')
    
    # 构建测试数据集
    test_data = []
    for item in split_info['test']:
        class_name = item['class_name']
        image_file = item['image_file']
        image_path = os.path.join(images_dir, class_name, image_file)
        
        # 获取类别标签
        label = list(ALGAE_DESCRIPTIONS.keys())[list(ALGAE_DESCRIPTIONS.values()).index(next(v for k, v in ALGAE_DESCRIPTIONS.items() if class_name in v))] if class_name in str(ALGAE_DESCRIPTIONS.values()) else 0
        
        test_data.append({
            'image_path': image_path,
            'label': label
        })
    
    print(f'测试数据集创建完成，包含 {len(test_data)} 张图像')
    return test_data

def load_model(model_path):
    """加载训练好的模型"""
    model = MultimodalModel()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f'模型加载成功: {model_path}')
    return model

def evaluate_model(model, test_data):
    """评估模型性能"""
    print('开始评估模型性能...')
    
    y_true = []
    y_pred = []
    y_prob = []
    
    from transformers import BertTokenizer
    import torch.nn.functional as F
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 生成类别文本嵌入
    class_texts = list(ALGAE_DESCRIPTIONS.values())
    class_embeddings = []
    
    model.eval()
    with torch.no_grad():
        for text in class_texts:
            inputs = tokenizer(
                text, 
                max_length=512, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
            input_ids = inputs['input_ids'].to(DEVICE)
            attention_mask = inputs['attention_mask'].to(DEVICE)
            
            _, text_emb = model(input_ids=input_ids, attention_mask=attention_mask)
            class_embeddings.append(text_emb.squeeze())
    
    class_embeddings = torch.stack(class_embeddings).to(DEVICE)
    
    # 处理测试数据
    from torchvision import transforms
    from PIL import Image
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model.eval()
    with torch.no_grad():
        for i, item in enumerate(test_data):
            if i % 50 == 0:
                print(f'处理测试样本 {i}/{len(test_data)}')
            
            # 加载图像
            try:
                image = Image.open(item['image_path']).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(DEVICE)
                
                # 获取图像嵌入
                img_embeds, _ = model(image=image_tensor)
                
                # 计算相似度
                similarities = F.cosine_similarity(img_embeds.unsqueeze(1), class_embeddings.unsqueeze(0), dim=2)
                
                # 获取预测结果
                pred_label = similarities.argmax(dim=1).item()
                pred_prob = similarities.max(dim=1)[0].item()
                
                # 保存结果
                y_true.append(item['label'])
                y_pred.append(pred_label)
                y_prob.append(pred_prob)
            except Exception as e:
                print(f'处理图像 {item["image_path"]} 时出错: {e}')
                continue
    
    return y_true, y_pred, y_prob

def calculate_metrics(y_true, y_pred):
    """计算性能指标"""
    print('计算性能指标...')
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print('\n性能指标:')
    print(f'准确率: {accuracy:.4f}')
    print(f'精确率: {precision:.4f}')
    print(f'召回率: {recall:.4f}')
    print(f'F1分数: {f1:.4f}')
    
    # 生成详细的分类报告
    print('\n分类报告:')
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_evaluation_results(metrics, y_true, y_pred, y_prob):
    """保存评估结果"""
    result_dir = os.path.join(project_root, 'results')
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存性能指标
    metrics_file = os.path.join(result_dir, 'model_evaluation_metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 保存预测结果
    predictions_file = os.path.join(result_dir, 'model_predictions.json')
    predictions = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f'评估结果已保存到: {metrics_file}')
    print(f'预测结果已保存到: {predictions_file}')
    
    return metrics_file, predictions_file

def generate_performance_visualization(metrics):
    """生成性能可视化"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制性能指标柱状图
    plt.figure(figsize=(10, 6))
    
    metrics_names = ['准确率', '精确率', '召回率', 'F1分数']
    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    
    bars = plt.bar(metrics_names, metrics_values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    plt.ylim(0, 1)
    plt.title('模型性能指标')
    plt.ylabel('分数')
    
    # 添加数值标签
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    # 保存图表
    result_dir = os.path.join(project_root, 'results')
    save_path = os.path.join(result_dir, 'model_performance_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'性能可视化已保存到: {save_path}')
    
    return save_path

def main():
    print('=== 模型评估工具 ===')
    
    # 1. 加载数据集划分结果
    split_info = load_dataset_split()
    if split_info is None:
        return
    
    # 2. 创建测试数据集
    test_data = create_test_dataset(split_info)
    
    # 3. 加载模型
    model_dir = os.path.join(project_root, 'models')
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if not model_files:
        print('错误：未找到训练好的模型文件')
        return
    
    print(f'找到 {len(model_files)} 个模型文件:')
    for i, model_file in enumerate(model_files):
        print(f'{i+1}. {model_file}')
    
    # 选择第一个模型进行评估
    model_path = os.path.join(model_dir, model_files[0])
    model = load_model(model_path)
    
    # 4. 评估模型
    y_true, y_pred, y_prob = evaluate_model(model, test_data)
    
    # 5. 计算性能指标
    metrics = calculate_metrics(y_true, y_pred)
    
    # 6. 保存评估结果
    save_evaluation_results(metrics, y_true, y_pred, y_prob)
    
    # 7. 生成性能可视化
    generate_performance_visualization(metrics)
    
    print('\n模型评估完成！')

if __name__ == '__main__':
    main()
