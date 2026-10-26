import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
from dataset import PneumoniaDataset
from model import PneumoniaModel
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score

# Focal Loss (handles imbalance) + Label Smoothing (handles noise)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        targets = targets.view(-1, 1)
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        BCE = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        return focal_loss.mean()

def main():
    # Load data
    df = pd.read_csv("../data/stage_1_train_labels.csv")
    df = df.drop_duplicates(subset='patientId')

    # Robust split: Group by patientId
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf.split(df, df['Target'], groups=df['patientId']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_ds = PneumoniaDataset(train_df, augment=True)
    val_ds = PneumoniaDataset(val_df, augment=False)

    # Reduced num_workers for Windows stability
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = PneumoniaModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = FocalLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_auc = 0
    for epoch in range(15):
        model.train()
        train_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            imgs, labels = imgs.cuda(), labels.cuda().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                imgs = imgs.cuda()
                outputs = model(imgs)
                preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                trues.extend(labels.numpy())
        
        auc = roc_auc_score(trues, preds)
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f} | Val AUC: {auc:.4f}")
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "../outputs/best_model.pth")
            print(f"New best model saved! AUC: {best_auc:.4f}")

    print("Training finished. Best model saved.")

if __name__ == '__main__':
    main()