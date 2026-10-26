import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve
from dataset import PneumoniaDataset
from model import PneumoniaModel
import numpy as np

def main():
    # Load best model
    model = PneumoniaModel()
    model.load_state_dict(torch.load("../outputs/best_model.pth", weights_only=True))
    model.cuda()
    model.eval()

    # Use a small hold-out set for quick evaluation
    df = pd.read_csv("../data/stage_1_train_labels.csv")
    df = df.drop_duplicates(subset='patientId')
    val_df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

    val_ds = PneumoniaDataset(val_df, augment=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print("Running evaluation on validation set...")

    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.cuda()
            outputs = model(imgs)
            preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            trues.extend(labels.numpy())

    preds = np.array(preds)
    trues = np.array(trues)

    # Clinical threshold (high sensitivity)
    threshold = 0.3
    pred_labels = (preds > threshold).astype(int)

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(trues, pred_labels, average='binary')
    auc = roc_auc_score(trues, preds)

    print("\n=== FINAL RESULTS ===")
    print(f"AUC          : {auc:.4f}")
    print(f"Sensitivity (Recall) : {recall:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print(f"Chosen threshold : {threshold} (for high sensitivity)")

    # Confusion Matrix
    cm = confusion_matrix(trues, pred_labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (threshold=0.3)')
    plt.savefig("../outputs/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(trues, preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("../outputs/roc_curve.png")
    plt.close()

    print("\n✅ All plots saved in outputs/ folder!")
    print("   - confusion_matrix.png")
    print("   - roc_curve.png")
    print("\nClinical Note: Threshold 0.3 chosen because missing pneumonia (False Negative) is very dangerous.")

if __name__ == '__main__':
    main()