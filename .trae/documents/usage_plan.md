# 项目使用脚本 - 详细步骤指南

## 任务 1: 数据准备和分割
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 运行数据集分割脚本，将藻类图像数据划分为训练集、验证集和测试集
- **Success Criteria**:
  - 生成 `data/dataset_split.json` 文件，包含训练集、验证集和测试集的划分结果
- **Test Requirements**:
  - `programmatic` TR-1.1: 运行脚本后，检查 `data/dataset_split.json` 文件是否生成
  - `programmatic` TR-1.2: 检查分割结果是否合理（训练集约70%，验证集约15%，测试集约15%）
- **使用脚本**:
  ```bash
  # 运行数据集分割脚本
  python scripts/dataset_splitter.py
  ```

## 任务 2: 模型训练
- **Priority**: P0
- **Depends On**: 任务 1
- **Description**:
  - 运行多模态模型训练脚本，使用训练集和验证集进行模型训练
- **Success Criteria**:
  - 在 `models/` 目录中生成训练好的模型文件
  - 训练过程中生成损失和准确率曲线
- **Test Requirements**:
  - `programmatic` TR-2.1: 运行脚本后，检查 `models/` 目录是否生成模型文件
  - `programmatic` TR-2.2: 检查训练日志是否显示训练过程正常
- **使用脚本**:
  ```bash
  # 运行模型训练脚本
  python src/train_multimodal.py
  ```

## 任务 3: 模型评估
- **Priority**: P0
- **Depends On**: 任务 2
- **Description**:
  - 运行模型评估脚本，使用测试集评估训练好的模型性能
- **Success Criteria**:
  - 生成模型性能指标（准确率、精确率、召回率、F1分数）
  - 生成分类报告和性能可视化图表
- **Test Requirements**:
  - `programmatic` TR-3.1: 运行脚本后，检查 `results/` 目录是否生成评估结果文件
  - `programmatic` TR-3.2: 检查评估结果是否显示模型性能良好
- **使用脚本**:
  ```bash
  # 运行模型评估脚本
  python scripts/model_evaluator.py
  ```

## 任务 4: 数据可视化
- **Priority**: P1
- **Depends On**: 任务 2
- **Description**:
  - 使用 `src/visualization.py` 中的功能进行数据可视化，包括文本注意力、热力图等
- **Success Criteria**:
  - 生成可视化结果，展示模型对图像和文本的理解
- **Test Requirements**:
  - `programmatic` TR-4.1: 运行可视化脚本后，检查是否生成可视化结果
  - `human-judgment` TR-4.2: 检查可视化结果是否清晰展示模型的注意力机制
- **使用脚本**:
  ```python
  # 在Python环境中使用可视化功能
  from src.visualization import visualize_text_attention, visualize_heatmap
  
  # 可视化文本注意力
  visualize_text_attention(model, text, image_path)
  
  # 可视化热力图
  visualize_heatmap(model, image_path)
  ```

## 任务 5: 零样本分类
- **Priority**: P1
- **Depends On**: 任务 2
- **Description**:
  - 使用训练好的模型进行零样本分类，识别未见过的藻类物种
- **Success Criteria**:
  - 模型能够基于文本描述对未见过的藻类图像进行分类
- **Test Requirements**:
  - `programmatic` TR-5.1: 运行零样本分类脚本后，检查分类结果是否合理
  - `human-judgment` TR-5.2: 检查分类结果是否符合预期
- **使用脚本**:
  ```python
  # 在Python环境中进行零样本分类
  from src.models.multimodal import MultimodalModel
  from transformers import BertTokenizer
  from torchvision import transforms
  from PIL import Image
  import torch
  
  # 加载模型
  model = MultimodalModel()
  model.load_state_dict(torch.load('models/multimodal_model_fold1.pth'))
  model.eval()
  
  # 准备图像和文本
  image_path = 'data/images/Anabaena/xxx.jpg'
  text_description = '这是一种蓝藻，具有丝状结构...'
  
  # 预处理图像
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  image = Image.open(image_path).convert('RGB')
  image_tensor = transform(image).unsqueeze(0)
  
  # 预处理文本
  tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
  inputs = tokenizer(text_description, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
  
  # 获取嵌入
  with torch.no_grad():
      img_emb, text_emb = model(image=image_tensor, input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
  
  # 计算相似度
  similarity = torch.cosine_similarity(img_emb, text_emb).item()
  print(f'图像与文本的相似度: {similarity}')
  ```

## 任务 6: 模型部署
- **Priority**: P2
- **Depends On**: 任务 2, 任务 3
- **Description**:
  - 将训练好的模型部署为服务，方便其他应用调用
- **Success Criteria**:
  - 创建一个简单的API服务，提供模型预测功能
- **Test Requirements**:
  - `programmatic` TR-6.1: 运行部署脚本后，检查服务是否正常启动
  - `programmatic` TR-6.2: 检查API是否能够正确返回预测结果
- **使用脚本**:
  ```python
  # 创建一个简单的Flask服务
  from flask import Flask, request, jsonify
  from src.models.multimodal import MultimodalModel
  from transformers import BertTokenizer
  from torchvision import transforms
  from PIL import Image
  import torch
  import os
  
  app = Flask(__name__)
  
  # 加载模型
  model = MultimodalModel()
  model.load_state_dict(torch.load('models/multimodal_model_fold1.pth'))
  model.eval()
  
  # 加载分词器
  tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
  
  # 图像预处理
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  
  @app.route('/predict', methods=['POST'])
  def predict():
      # 获取图像和文本
      image_file = request.files['image']
      text = request.form['text']
      
      # 预处理图像
      image = Image.open(image_file).convert('RGB')
      image_tensor = transform(image).unsqueeze(0)
      
      # 预处理文本
      inputs = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
      
      # 获取嵌入
      with torch.no_grad():
          img_emb, text_emb = model(image=image_tensor, input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
      
      # 计算相似度
      similarity = torch.cosine_similarity(img_emb, text_emb).item()
      
      return jsonify({'similarity': similarity})
  
  if __name__ == '__main__':
      app.run(host='0.0.0.0', port=5000)
  ```

## 任务 7: 模型调优
- **Priority**: P2
- **Depends On**: 任务 2, 任务 3
- **Description**:
  - 基于评估结果，调整模型参数，提高模型性能
- **Success Criteria**:
  - 模型性能得到提升
- **Test Requirements**:
  - `programmatic` TR-7.1: 运行调优后的模型，检查性能是否提升
  - `human-judgment` TR-7.2: 分析调优前后的性能差异
- **使用脚本**:
  ```bash
  # 修改配置文件中的参数
  # 例如，调整学习率、批次大小等
  # 然后重新训练模型
  python src/train_multimodal.py
  
  # 评估调优后的模型
  python scripts/model_evaluator.py
  ```

## 任务 8: 数据增强
- **Priority**: P2
- **Depends On**: 任务 1
- **Description**:
  - 使用数据增强技术，扩充训练数据集，提高模型的泛化能力
- **Success Criteria**:
  - 生成增强后的训练数据
  - 模型在测试集上的性能得到提升
- **Test Requirements**:
  - `programmatic` TR-8.1: 运行数据增强脚本后，检查增强数据是否生成
  - `programmatic` TR-8.2: 训练使用增强数据的模型，检查性能是否提升
- **使用脚本**:
  ```python
  # 在 dataset.py 中添加数据增强功能
  # 例如，添加随机裁剪、翻转、旋转等操作
  # 然后重新训练模型
  python src/train_multimodal.py
  ```

## 总结

通过以上步骤，您可以完成从数据准备到模型训练、评估、可视化和部署的完整流程。每个步骤都有详细的脚本和说明，您可以根据实际需求选择执行相应的任务。

### 推荐使用流程
1. **数据准备和分割** (任务 1)
2. **模型训练** (任务 2)
3. **模型评估** (任务 3)
4. **数据可视化** (任务 4)
5. **零样本分类** (任务 5)
6. **模型部署** (任务 6)
7. **模型调优** (任务 7)
8. **数据增强** (任务 8)

这样的流程可以帮助您充分利用项目的功能，实现藻类识别的目标。