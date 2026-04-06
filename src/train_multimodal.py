# 双塔模型训练与零样本验证逻辑

import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import (
    DEVICE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, EPOCHS, LR, WEIGHT_DECAY,
    USE_AMP, N_FOLDS, NUM_UNSEEN_SPECIES, ALGAE_DESCRIPTIONS, MODEL_SAVE_DIR
)
from dataset import AlgaeMultimodalDataset, get_stratified_kfold, get_data_loaders
from models.multimodal import MultimodalModel
from loss import InfoNCELoss
from transformers import BertTokenizer

def train_epoch(model, train_loader, optimizer, loss_fn, scaler, gradient_accumulation_steps):
    """
    训练一个epoch
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        loss_fn: 损失函数
        scaler: 梯度缩放器
        gradient_accumulation_steps: 梯度累加步数
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0
    step = 0

    for batch in tqdm(train_loader, desc='Training'):
        image_tensor, input_ids, attention_mask, _ = batch
        image_tensor = image_tensor.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        with autocast(device_type='cuda', enabled=USE_AMP):
            # 前向传播
            image_embeds, text_embeds = model(image_tensor, input_ids, attention_mask)
            # 计算损失
            loss = loss_fn(image_embeds, text_embeds)
            # 梯度累加
            loss = loss / gradient_accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()
        total_loss += loss.item() * gradient_accumulation_steps
        step += 1

        # 每gradient_accumulation_steps步更新一次参数
        if step % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return total_loss / len(train_loader)

def zero_shot_validate(model, val_loader, tokenizer, unseen_species):
    """
    零样本验证
    Args:
        model: 模型
        val_loader: 验证数据加载器
        tokenizer: BERT分词器
        unseen_species: 未见过的物种ID列表
    Returns:
        准确率和F1-Score
    """
    model.eval()
    all_preds = []
    all_labels = []

    # 准备文本描述嵌入
    text_embeds_list = []
    for label, desc in ALGAE_DESCRIPTIONS.items():
        # 处理文本
        text_inputs = tokenizer(
            desc,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = text_inputs['input_ids'].to(DEVICE)
        attention_mask = text_inputs['attention_mask'].to(DEVICE)

        # 获取文本嵌入
        with torch.no_grad():
            _, text_emb = model(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds_list.append(text_emb)

    # 合并文本嵌入
    text_embeds = torch.cat(text_embeds_list, dim=0)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Zero-shot Validation'):
            image_tensor, _, _, label_id = batch
            image_tensor = image_tensor.to(DEVICE)
            label_id = label_id.cpu().numpy()

            # 过滤出未见过的物种
            unseen_mask = np.isin(label_id, unseen_species)
            if not np.any(unseen_mask):
                continue

            # 获取图像嵌入
            image_embeds, _ = model(image=image_tensor[unseen_mask])

            # 计算相似度
            similarity = torch.matmul(image_embeds, text_embeds.t())
            # 预测标签
            preds = torch.argmax(similarity, dim=1).cpu().numpy()

            # 收集预测和真实标签
            all_preds.extend(preds)
            all_labels.extend(label_id[unseen_mask])

    # 计算指标
    if len(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
    else:
        accuracy = 0.0
        f1 = 0.0

    return accuracy, f1

def main():
    """主函数"""
    # 创建模型保存目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 加载数据集
    dataset = AlgaeMultimodalDataset(os.path.join(project_root, 'data'))

    # 交叉验证
    print(f"使用 {N_FOLDS} 折交叉验证")
    folds = get_stratified_kfold(dataset.data, n_splits=N_FOLDS)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n=== Fold {fold_idx + 1}/{N_FOLDS} ===")

        # 分割数据
        train_data = [dataset.data[i] for i in train_idx]
        val_data = [dataset.data[i] for i in val_idx]

        # 提取未见过的物种（从验证集中随机选择2个）
        val_labels = [item['label'] for item in val_data]
        unique_labels = list(set(val_labels))
        if len(unique_labels) >= NUM_UNSEEN_SPECIES:
            unseen_species = np.random.choice(unique_labels, NUM_UNSEEN_SPECIES, replace=False)
        else:
            unseen_species = unique_labels

        print(f"Unseen species: {unseen_species}")

        # 创建数据加载器
        # 这里简化处理，实际项目中可能需要创建临时目录或使用Subset
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )

        # 初始化模型
        model = MultimodalModel().to(DEVICE)

        # 初始化优化器
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        # 初始化损失函数
        loss_fn = InfoNCELoss()

        # 初始化梯度缩放器
        scaler = GradScaler(enabled=USE_AMP)

        # 训练循环
        best_f1 = 0.0

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")

            # 训练
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_fn, scaler, GRADIENT_ACCUMULATION_STEPS
            )
            print(f"Train Loss: {train_loss:.4f}")

            # 零样本验证
            accuracy, f1 = zero_shot_validate(model, val_loader, tokenizer, unseen_species)
            print(f"Zero-shot Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

            # 保存最佳模型
            if f1 > best_f1:
                best_f1 = f1
                model_path = os.path.join(MODEL_SAVE_DIR, f'multimodal_model_fold{fold_idx + 1}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model to {model_path}")

        print(f"\nFold {fold_idx + 1} completed. Best F1-Score: {best_f1:.4f}")

if __name__ == '__main__':
    main()
