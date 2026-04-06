# 数据集划分脚本
# 按70%:15%:15%比例划分数据集为训练集、验证集和测试集

import os
import sys
import json
import random
from sklearn.model_selection import train_test_split

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 设置随机种子，确保可重复性
random.seed(42)

def analyze_dataset_structure():
    """分析当前数据集结构"""
    data_dir = os.path.join(project_root, 'data')
    images_dir = os.path.join(data_dir, 'images')
    
    print('分析数据集结构...')
    
    # 遍历所有类别目录
    classes = []
    total_images = 0
    
    if not os.path.exists(images_dir):
        print(f'错误：{images_dir} 目录不存在')
        return []
    
    for class_name in os.listdir(images_dir):
        class_dir = os.path.join(images_dir, class_name)
        if os.path.isdir(class_dir):
            # 统计该类别的图像数量
            image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            num_images = len(image_files)
            classes.append({
                'class_name': class_name,
                'num_images': num_images,
                'image_files': image_files
            })
            total_images += num_images
            print(f'类别 {class_name}: {num_images} 张图像')
    
    print(f'\n总图像数: {total_images}')
    print(f'总类别数: {len(classes)}')
    
    return classes

def split_dataset(classes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """划分数据集"""
    print('\n开始划分数据集...')
    
    # 确保比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 存储划分结果
    train_data = []
    val_data = []
    test_data = []
    
    # 对每个类别单独划分
    for cls in classes:
        class_name = cls['class_name']
        image_files = cls['image_files']
        num_images = cls['num_images']
        
        # 第一次划分：训练集和剩余部分
        train_files, remaining_files = train_test_split(
            image_files, 
            train_size=train_ratio, 
            random_state=42
        )
        
        # 计算验证集和测试集的比例
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        
        # 第二次划分：验证集和测试集
        val_files, test_files = train_test_split(
            remaining_files, 
            train_size=val_test_ratio, 
            random_state=42
        )
        
        # 添加到对应的数据集中
        for file in train_files:
            train_data.append({
                'class_name': class_name,
                'image_file': file
            })
        
        for file in val_files:
            val_data.append({
                'class_name': class_name,
                'image_file': file
            })
        
        for file in test_files:
            test_data.append({
                'class_name': class_name,
                'image_file': file
            })
        
        print(f'类别 {class_name} 划分完成: 训练集 {len(train_files)}, 验证集 {len(val_files)}, 测试集 {len(test_files)}')
    
    # 打印总体划分结果
    print('\n总体划分结果:')
    print(f'训练集: {len(train_data)} 张图像 ({len(train_data)/sum([c["num_images"] for c in classes])*100:.1f}%)')
    print(f'验证集: {len(val_data)} 张图像 ({len(val_data)/sum([c["num_images"] for c in classes])*100:.1f}%)')
    print(f'测试集: {len(test_data)} 张图像 ({len(test_data)/sum([c["num_images"] for c in classes])*100:.1f}%)')
    
    return train_data, val_data, test_data

def save_split_results(train_data, val_data, test_data):
    """保存划分结果"""
    data_dir = os.path.join(project_root, 'data')
    
    # 保存为JSON文件
    split_info = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    split_file = os.path.join(data_dir, 'dataset_split.json')
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    print(f'\n划分结果已保存到: {split_file}')
    
    return split_file

def generate_split_visualization(classes, train_data, val_data, test_data):
    """生成数据划分可视化"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 数据划分比例饼图
    plt.figure(figsize=(12, 6))
    
    # 饼图
    plt.subplot(1, 2, 1)
    sizes = [len(train_data), len(val_data), len(test_data)]
    labels = ['训练集', '验证集', '测试集']
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('数据划分比例')
    
    # 2. 各类别在不同数据集的分布
    plt.subplot(1, 2, 2)
    
    # 统计每个类别的分布
    class_distribution = {}
    for cls in classes:
        class_name = cls['class_name']
        class_distribution[class_name] = {
            'train': sum(1 for item in train_data if item['class_name'] == class_name),
            'val': sum(1 for item in val_data if item['class_name'] == class_name),
            'test': sum(1 for item in test_data if item['class_name'] == class_name)
        }
    
    # 绘制堆叠柱状图
    class_names = list(class_distribution.keys())
    train_counts = [class_distribution[c]['train'] for c in class_names]
    val_counts = [class_distribution[c]['val'] for c in class_names]
    test_counts = [class_distribution[c]['test'] for c in class_names]
    
    x = range(len(class_names))
    width = 0.35
    
    plt.bar(x, train_counts, width, label='训练集', color='#4CAF50')
    plt.bar(x, val_counts, width, bottom=train_counts, label='验证集', color='#2196F3')
    plt.bar(x, test_counts, width, bottom=[i+j for i,j in zip(train_counts, val_counts)], label='测试集', color='#FF9800')
    
    plt.xlabel('类别')
    plt.ylabel('图像数量')
    plt.title('各类别在不同数据集的分布')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    result_dir = os.path.join(project_root, 'results')
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, 'dataset_split_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'数据划分可视化已保存到: {save_path}')
    
    return save_path

def main():
    print('=== 数据集划分工具 ===')
    
    # 1. 分析数据集结构
    classes = analyze_dataset_structure()
    
    if not classes:
        print('错误：未找到数据集')
        return
    
    # 2. 划分数据集
    train_data, val_data, test_data = split_dataset(classes)
    
    # 3. 保存划分结果
    save_split_results(train_data, val_data, test_data)
    
    # 4. 生成可视化
    generate_split_visualization(classes, train_data, val_data, test_data)
    
    print('\n数据集划分完成！')

if __name__ == '__main__':
    main()
