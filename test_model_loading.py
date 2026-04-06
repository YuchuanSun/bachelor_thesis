# 测试模型加载和Grad-CAM生成

import os
import sys
import torch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.models.multimodal import MultimodalModel
from src.visualization import VisualizationTool
from src.dataset import AlgaeMultimodalDataset
from torch.utils.data import DataLoader

from configs.default_config import DEVICE, BATCH_SIZE

def test_model_loading():
    """测试模型加载"""
    print("测试模型加载...")
    
    # 检查模型文件是否存在
    model_dir = os.path.join(project_root, 'models')
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if not model_files:
        print("错误: 没有找到模型文件")
        return False
    
    print(f"找到 {len(model_files)} 个模型文件:")
    for model_file in model_files:
        print(f"- {model_file}")
    
    # 加载模型
    model_path = os.path.join(model_dir, model_files[0])
    print(f"\n加载模型: {model_path}")
    
    try:
        model = MultimodalModel()
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("模型加载成功!")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return False

def test_grad_cam(model):
    """测试Grad-CAM生成"""
    if not model:
        print("错误: 模型未加载")
        return False
    
    print("\n测试Grad-CAM生成...")
    
    # 获取测试数据
    dataset = AlgaeMultimodalDataset(data_dir=os.path.join(project_root, 'data'), split='test')
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0
    )
    
    # 获取一个样本
    for batch in dataloader:
        image = batch[0].to(DEVICE)
        label = batch[3].item()
        print(f"使用样本标签: {label}")
        break
    
    # 生成Grad-CAM热力图
    viz_tool = VisualizationTool()
    save_path = os.path.join(project_root, 'results', 'test_grad_cam.png')
    
    try:
        grad_cam_path = viz_tool.generate_grad_cam(model, image, class_idx=label, save_path=save_path)
        if grad_cam_path:
            print(f"Grad-CAM热力图生成成功: {grad_cam_path}")
            return True
        else:
            print("Grad-CAM热力图生成失败")
            return False
    except Exception as e:
        print(f"生成Grad-CAM失败: {e}")
        return False

def main():
    """主函数"""
    print("开始测试模型加载和Grad-CAM生成...")
    
    # 测试模型加载
    model = test_model_loading()
    
    # 测试Grad-CAM生成
    if model:
        test_grad_cam(model)
    
    print("\n测试完成!")

if __name__ == '__main__':
    main()
