import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from datetime import datetime
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import (
    DEVICE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, EPOCHS, LR, WEIGHT_DECAY,
    USE_AMP, N_FOLDS, NUM_UNSEEN_SPECIES, ALGAE_DESCRIPTIONS, MODEL_SAVE_DIR, RESULT_DIR
)
from dataset import AlgaeMultimodalDataset, get_stratified_kfold, get_data_loaders
from models.multimodal import MultimodalModel
from loss import InfoNCELoss
from transformers import BertTokenizer

def train_epoch(model, train_loader, optimizer, loss_fn, scaler, gradient_accumulation_steps):
    model.train()
    total_loss = 0
    step = 0

    for batch in tqdm(train_loader, desc='Training'):
        image_tensor, input_ids, attention_mask, _ = batch
        image_tensor = image_tensor.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        with autocast(device_type='cuda', enabled=USE_AMP):
            image_embeds, text_embeds = model(image_tensor, input_ids, attention_mask)
            loss = loss_fn(image_embeds, text_embeds)
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * gradient_accumulation_steps
        step += 1

        if step % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return total_loss / len(train_loader)

def zero_shot_validate(model, val_loader, tokenizer, unseen_species):
    model.eval()
    all_preds = []
    all_labels = []

    text_embeds_list = []
    for label, desc in ALGAE_DESCRIPTIONS.items():
        text_inputs = tokenizer(
            desc,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = text_inputs['input_ids'].to(DEVICE)
        attention_mask = text_inputs['attention_mask'].to(DEVICE)

        with torch.no_grad():
            _, text_emb = model(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds_list.append(text_emb)

    text_embeds = torch.cat(text_embeds_list, dim=0)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Zero-shot Validation'):
            image_tensor, _, _, label_id = batch
            image_tensor = image_tensor.to(DEVICE)
            label_id = label_id.cpu().numpy()

            unseen_mask = np.isin(label_id, unseen_species)
            if not np.any(unseen_mask):
                continue

            image_embeds_batch, _ = model(image=image_tensor[unseen_mask])
            similarity = torch.matmul(image_embeds_batch, text_embeds.t())
            preds = torch.argmax(similarity, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(label_id[unseen_mask])

    if len(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
    else:
        accuracy = 0.0
        f1 = 0.0

    return accuracy, f1

def generate_zeroshot_visualization(model, sample_image_path, tokenizer, species_names, save_path):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    texts = [ALGAE_DESCRIPTIONS[i] for i in range(len(ALGAE_DESCRIPTIONS))]
    
    raw_img = Image.open(sample_image_path).convert('RGB')
    img_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_feat = model(image=img_tensor)[0]
        img_feat = F.normalize(img_feat, p=2, dim=1)
        
        text_inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        txt_output = model.text_encoder(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'])
        txt_feat = model.text_proj(txt_output.pooler_output)
        txt_feat = F.normalize(txt_feat, p=2, dim=1)
        
        similarities = torch.matmul(img_feat, txt_feat.T).squeeze(0).cpu().numpy()

    sorted_indices = similarities.argsort()[-5:][::-1]
    top_species = [species_names[i] for i in sorted_indices]
    top_scores = [similarities[i] for i in sorted_indices]

    colors = ['#e74c3c' if i == 0 else '#95a5a6' for i in range(len(top_scores))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.5]})
    
    ax1.imshow(raw_img)
    ax1.axis('off')
    ax1.set_title(f"Zero-shot Input Image\nTrue Label: {species_names[sorted_indices[0]]}", fontsize=14, fontweight='bold')
    
    bars = ax2.barh(top_species[::-1], top_scores[::-1], color=colors[::-1])
    ax2.set_xlabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Text Dictionary Retrieval Results (Top-5)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1.0)
    
    for bar in bars:
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.3f}', va='center', fontsize=11)
                
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zero-shot visualization saved to: {save_path}")

def export_training_results_to_excel(results, filename):
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    print(f"Training results exported to: {filename}")

def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset = AlgaeMultimodalDataset(os.path.join(project_root, 'data'))
    species_names = [f"Species_{i}" for i in range(len(ALGAE_DESCRIPTIONS))]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = os.path.join(RESULT_DIR, f"training_results_{timestamp}.xlsx")

    all_training_results = []

    print(f"Using {N_FOLDS}-fold cross-validation")
    folds = get_stratified_kfold(dataset.data, n_splits=N_FOLDS)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n=== Fold {fold_idx + 1}/{N_FOLDS} ===")

        train_data = [dataset.data[i] for i in train_idx]
        val_data = [dataset.data[i] for i in val_idx]

        val_labels = [item['label'] for item in val_data]
        unique_labels = list(set(val_labels))
        if len(unique_labels) >= NUM_UNSEEN_SPECIES:
            unseen_species = np.random.choice(unique_labels, NUM_UNSEEN_SPECIES, replace=False)
        else:
            unseen_species = unique_labels

        print(f"Unseen species: {unseen_species}")

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

        model = MultimodalModel().to(DEVICE)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        loss_fn = InfoNCELoss()
        scaler = GradScaler(enabled=USE_AMP)

        best_f1 = 0.0
        best_accuracy = 0.0
        fold_results = []

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")

            train_loss = train_epoch(
                model, train_loader, optimizer, loss_fn, scaler, GRADIENT_ACCUMULATION_STEPS
            )
            print(f"Train Loss: {train_loss:.4f}")

            accuracy, f1 = zero_shot_validate(model, val_loader, tokenizer, unseen_species)
            print(f"Zero-shot Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

            fold_results.append({
                'Fold': fold_idx + 1,
                'Epoch': epoch + 1,
                'Train_Loss': train_loss,
                'Zero_shot_Accuracy': accuracy,
                'Zero_shot_F1': f1
            })

            if f1 > best_f1:
                best_f1 = f1
                best_accuracy = accuracy
                model_path = os.path.join(MODEL_SAVE_DIR, f'multimodal_model_fold{fold_idx + 1}_{timestamp}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model to {model_path}")

        print(f"\nFold {fold_idx + 1} completed. Best F1-Score: {best_f1:.4f}, Best Accuracy: {best_accuracy:.4f}")

        all_training_results.extend(fold_results)

        viz_save_path = os.path.join(RESULT_DIR, f'zeroshot_visualization_fold{fold_idx + 1}_{timestamp}.png')
        sample_images = [item['image_path'] for item in val_data if item['label'] in unseen_species]
        if sample_images:
            generate_zeroshot_visualization(model, sample_images[0], tokenizer, species_names, viz_save_path)

        fold_summary = {
            'Model': f'multimodal_model_fold{fold_idx + 1}_{timestamp}',
            'Fold': fold_idx + 1,
            'Best_F1_Score': best_f1,
            'Best_Accuracy': best_accuracy,
            'Unseen_Species': str(list(unseen_species)),
            'Total_Parameters': sum(p.numel() for p in model.parameters()),
            'Trainable_Parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        all_training_results.append(fold_summary)

    export_training_results_to_excel(all_training_results, excel_filename)

    print(f"\n=== Training Complete ===")
    print(f"All results saved to: {excel_filename}")

if __name__ == '__main__':
    main()
