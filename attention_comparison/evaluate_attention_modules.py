# -*- coding: utf-8 -*-
# evaluate_attention_modules.py
# Attention Ablation Study with Two-Level Leave-One-Well-Out

import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ==============================================================================
# Configuration
# ==============================================================================

SEED = 42
DATA_DIR = "vdl_slices_20px"
BATCH_SIZE = 32
NUM_CLASSES = 3
LEARNING_RATE = 2e-5
IMAGE_SIZE = (500, 40)
INNER_EPOCHS = 50
USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHT_IN_LOSS = False
NUM_WORKERS = 4

# Attention Configuration - Change this to run ablation experiments
ATTENTION = "ASA"  # Options: "ASA", "SE", "ECA", "CBAM", None

label_map = {"Good": 0, "Midrate": 1, "Poor": 2}
label_names = ["Good", "Midrate", "Poor"]

# ==============================================================================
# Helper Functions
# ==============================================================================


def is_valid_well_dir(root_dir, well_name):
    """Filter non-well folders such as hidden dirs and notebook checkpoints."""
    well_path = os.path.join(root_dir, well_name)
    if not os.path.isdir(well_path):
        return False
    if well_name.startswith("."):
        return False
    if "checkpoint" in well_name.lower():
        return False
    if well_name == "__pycache__":
        return False
    return True


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Dataset (supports filtering by well)
# ==============================================================================


class CementVDLDataset(Dataset):
    def __init__(self, root_dir, transform=None, selected_wells=None):
        self.samples = []
        self.transform = transform

        all_wells = sorted(
            [w for w in os.listdir(root_dir) if is_valid_well_dir(root_dir, w)]
        )

        if selected_wells is None:
            wells = all_wells
        else:
            selected_set = set(selected_wells)
            wells = [w for w in all_wells if w in selected_set]

        for well in wells:
            well_dir = os.path.join(root_dir, well)
            for label_name, label_idx in label_map.items():
                class_dir = os.path.join(well_dir, label_name)
                if not os.path.isdir(class_dir):
                    continue

                for fname in os.listdir(class_dir):
                    if fname.endswith(".png"):
                        self.samples.append((os.path.join(class_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label


# ==============================================================================
# Attention Modules
# ==============================================================================


class AnisotropicSpatialAttention(nn.Module):
    """
    各向异性空间注意力模块 (Anisotropic Spatial Attention, ASA)
    专为高宽比（条状）特征图设计
    """

    def __init__(self, in_planes):
        super(AnisotropicSpatialAttention, self).__init__()
        # 使用大的1D卷积核捕捉长距离水平特征
        self.horizontal_conv = nn.Conv1d(
            in_planes, in_planes, kernel_size=21, padding=10, groups=in_planes, bias=False)
        # 使用小的1D卷积核捕捉短距离垂直特征
        self.vertical_conv = nn.Conv1d(
            in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes, bias=False)
        # 融合两个方向的注意力图
        self.fusion_conv = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, C, H, W]

        # 1. 水平注意力分支
        x_h_pooled = x.mean(dim=2)  # 高度池化 -> [B, C, W]
        h_attn = self.horizontal_conv(x_h_pooled)
        # 扩展回原高度 -> [B, C, H, W]
        h_attn = h_attn.unsqueeze(2).expand(-1, -1, x.shape[2], -1)

        # 2. 垂直注意力分支
        x_v_pooled = x.mean(dim=3)  # 宽度池化 -> [B, C, H]
        v_attn = self.vertical_conv(x_v_pooled)  # Conv1d期望输入是 [B, C, L]
        # 扩展回原宽度 -> [B, C, H, W]
        v_attn = v_attn.unsqueeze(3).expand(-1, -1, -1, x.shape[3])

        # 3. 融合两个方向的注意力
        fused_attn = h_attn + v_attn

        # 4. 生成最终的单通道空间注意力图 (类似CBAM)
        avg_out = torch.mean(fused_attn, dim=1, keepdim=True)
        max_out, _ = torch.max(fused_attn, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.fusion_conv(attention_map)

        return self.sigmoid(attention_map)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(),
            nn.Linear(channels // r, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b,c,_,_ = x.shape
        y = self.pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y


class ECABlock(nn.Module):
    """Efficient Channel Attention Block"""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=k_size,padding=(k_size-1)//2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        y = self.pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2))
        y = self.sigmoid(y).transpose(-1,-2).unsqueeze(-1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, r=16):
        super().__init__()

        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels//r),
            nn.ReLU(),
            nn.Linear(channels//r, channels)
        )

        self.spatial = nn.Conv2d(2,1,kernel_size=7,padding=3)

    def forward(self,x):

        b,c,h,w = x.shape

        avg = torch.mean(x,(2,3))
        mx = torch.amax(x,(2,3))

        ca = torch.sigmoid(self.channel_mlp(avg)+self.channel_mlp(mx)).view(b,c,1,1)
        x = x * ca

        avg = torch.mean(x,1,keepdim=True)
        mx,_ = torch.max(x,1,keepdim=True)

        sa = torch.sigmoid(self.spatial(torch.cat([avg,mx],1)))

        return x * sa


# ==============================================================================
# Attention Factory Function
# ==============================================================================


def build_attention(att_type, channels):
    """Factory function to build attention modules based on type"""
    if att_type == "ASA":
        return AnisotropicSpatialAttention(channels)
    if att_type == "SE":
        return SEBlock(channels)
    if att_type == "ECA":
        return ECABlock(channels)
    if att_type == "CBAM":
        return CBAM(channels)
    return nn.Identity()  # Baseline: no attention


# ==============================================================================
# Model
# ==============================================================================


class LightweightVDLNet_PlacementAblation(nn.Module):
    def __init__(self, num_classes=3, attention="ASA", asa_before_pool=True):
        super(LightweightVDLNet_PlacementAblation, self).__init__()
        self.attention = attention
        self.asa_before_pool = asa_before_pool

        # --- Block 1 ---
        self.block1_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.block1_pool = nn.MaxPool2d(2)

        # --- Block 2 ---
        self.block2_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.block2_pool = nn.MaxPool2d(2)

        # --- Block 3 ---
        self.block3_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.block3_pool = nn.MaxPool2d(2)

        # Attention modules - using factory pattern
        self.att1 = build_attention(self.attention, 32)
        self.att2 = build_attention(self.attention, 64)
        self.att3 = build_attention(self.attention, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def apply_attention(self, x, att_module):
        """Apply attention module uniformly.
        
        For ASA: returns attention map [B,1,H,W], then applies residual weighting: x + x * attn
        For SE/ECA/CBAM/Identity: returns weighted feature map directly [B,C,H,W]
        """
        if self.attention == "ASA":
            attn = att_module(x)  # [B,1,H,W] attention map
            return x + x * attn.expand_as(x)  # residual ASA
        else:
            return att_module(x)  # SE/ECA/CBAM/Identity - returns weighted features

    def forward(self, x):
        # --- Block 1 ---
        x = self.block1_conv(x)
        if self.asa_before_pool:
            x = self.apply_attention(x, self.att1)
        x = self.block1_pool(x)
        if not self.asa_before_pool:
            x = self.apply_attention(x, self.att1)

        # --- Block 2 ---
        x = self.block2_conv(x)
        if self.asa_before_pool:
            x = self.apply_attention(x, self.att2)
        x = self.block2_pool(x)
        if not self.asa_before_pool:
            x = self.apply_attention(x, self.att2)

        # --- Block 3 ---
        x = self.block3_conv(x)
        if self.asa_before_pool:
            x = self.apply_attention(x, self.att3)
        x = self.block3_pool(x)
        if not self.asa_before_pool:
            x = self.apply_attention(x, self.att3)

        # --- Classifier ---
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ==============================================================================
# Data / Loss / Train Helpers
# ==============================================================================


def get_transform():
    return transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])


def get_all_wells(root_dir):
    wells = sorted([w for w in os.listdir(root_dir) if is_valid_well_dir(root_dir, w)])
    return wells


def build_loader(dataset, is_train):
    if is_train and USE_WEIGHTED_SAMPLER:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        classes, counts = np.unique(labels, return_counts=True)
        class_weights_for_sampler = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
        sample_weights = np.array([class_weights_for_sampler[l] for l in labels])
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=is_train,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def compute_class_weights(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    # align to full class index [0, 1, 2]
    full_weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for cls_idx, cls_weight in zip(classes, weights):
        full_weights[int(cls_idx)] = cls_weight
    return torch.tensor(full_weights, dtype=torch.float32, device=device)


def evaluate(model, loader, criterion_cls):
    """Evaluate model on given dataloader and return metrics."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion_cls(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    acc = 100.0 * (np.array(all_preds) == np.array(all_labels)).mean() if all_labels else 0.0
    return {
        "loss": avg_loss,
        "acc": acc,
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
    }


def train_one_fold(train_loader, val_loader, fixed_epochs, attention_type):
    """Train model for a fixed number of epochs (no early stopping)."""
    model = LightweightVDLNet_PlacementAblation(
        num_classes=NUM_CLASSES,
        attention=attention_type,
        asa_before_pool=True,
    ).to(device)

    class_weights = compute_class_weights(train_loader.dataset) if USE_CLASS_WEIGHT_IN_LOSS else None
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-10)

    epoch_metrics = []

    for epoch in range(fixed_epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{fixed_epochs}", leave=False)
        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion_cls(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(train_loss=f"{train_loss / max(1, len(train_loader)):.4f}")

        val_result = evaluate(model, val_loader, criterion_cls)
        scheduler.step(val_result["loss"])

        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "val_loss": val_result["loss"],
                "val_acc": val_result["acc"],
            }
        )

    return {
        "epoch_metrics": epoch_metrics,
    }


def inner_loo_select_best_epoch(outer_train_wells, transform, fixed_epochs, attention_type):
    """Select best epoch using inner leave-one-well-out."""
    inner_histories = []

    for inner_val_well in outer_train_wells:
        inner_train_wells = [w for w in outer_train_wells if w != inner_val_well]

        train_ds = CementVDLDataset(DATA_DIR, transform=transform, selected_wells=inner_train_wells)
        val_ds = CementVDLDataset(DATA_DIR, transform=transform, selected_wells=[inner_val_well])

        if len(train_ds) == 0 or len(val_ds) == 0:
            raise RuntimeError(
                f"Inner fold has empty dataset. train={inner_train_wells}, val={[inner_val_well]}"
            )

        train_loader = build_loader(train_ds, is_train=True)
        val_loader = build_loader(val_ds, is_train=False)

        print(f"  Inner fold: train={inner_train_wells}, val={[inner_val_well]}")
        fold_result = train_one_fold(train_loader, val_loader, fixed_epochs=fixed_epochs, attention_type=attention_type)
        inner_histories.append(pd.DataFrame(fold_result["epoch_metrics"]))

    merged = inner_histories[0][["epoch"]].copy()
    merged["mean_val_acc"] = 0.0
    merged["mean_val_loss"] = 0.0

    for hist in inner_histories:
        merged["mean_val_acc"] += hist["val_acc"]
        merged["mean_val_loss"] += hist["val_loss"]

    merged["mean_val_acc"] /= len(inner_histories)
    merged["mean_val_loss"] /= len(inner_histories)

    # primary criterion: highest mean val acc; tie-breaker: lower mean val loss
    best_row = merged.sort_values(["mean_val_acc", "mean_val_loss"], ascending=[False, True]).iloc[0]
    best_epoch = int(best_row["epoch"])

    return best_epoch, merged


def plot_precision_recall_curve(y_true, y_probs, num_classes, label_names, title, save_path):
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        y_true_class = (np.array(y_true) == i).astype(int)
        y_score_class = np.array(y_probs)[:, i]

        precision, recall, _ = precision_recall_curve(
            y_true_class, y_score_class)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2,
                 label=f'Class {label_names[i]} (AUC = {pr_auc:.2f})')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def train_full_outer_and_test(outer_train_wells, outer_test_well, transform, best_epoch, attention_type):
    """Train final model on all outer_train_wells and test on outer_test_well."""
    train_ds = CementVDLDataset(DATA_DIR, transform=transform, selected_wells=outer_train_wells)
    test_ds = CementVDLDataset(DATA_DIR, transform=transform, selected_wells=[outer_test_well])

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            f"Outer fold has empty dataset. train={outer_train_wells}, test={[outer_test_well]}"
        )

    train_loader = build_loader(train_ds, is_train=True)
    test_loader = build_loader(test_ds, is_train=False)

    # final training: no validation split, train exactly best_epoch rounds
    model = LightweightVDLNet_PlacementAblation(
        num_classes=NUM_CLASSES,
        attention=attention_type,
        asa_before_pool=True,
    ).to(device)

    class_weights = compute_class_weights(train_ds) if USE_CLASS_WEIGHT_IN_LOSS else None
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    print(f"  Final outer training for {best_epoch} epochs on wells={outer_train_wells}")
    for epoch in range(best_epoch):
        model.train()
        loop = tqdm(train_loader, desc=f"Final train epoch {epoch + 1}/{best_epoch}", leave=False)

        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion_cls(outputs, labels)
            loss.backward()
            optimizer.step()

    # 保存模型
    model_path = f"model_outer_{outer_test_well}_{attention_type}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model: {model_path}")

    test_result = evaluate(model, test_loader, criterion_cls)
    report_text = classification_report(
        test_result["labels"],
        test_result["preds"],
        target_names=label_names,
        zero_division=0,
        digits=5,
    )
    cm = confusion_matrix(test_result["labels"], test_result["preds"], labels=[0, 1, 2])

    # 保存分类报告
    report_path = f"classification_report_test_outer_{outer_test_well}_{attention_type}.csv"
    report_dict = classification_report(
        test_result["labels"],
        test_result["preds"],
        target_names=label_names,
        output_dict=True,
        zero_division=0,
        digits=5,
    )
    pd.DataFrame(report_dict).transpose().to_csv(report_path, index=True)
    print(f"Saved classification report: {report_path}")

    # 保存混淆矩阵
    cm_path = f"confusion_matrix_test_outer_{outer_test_well}_{attention_type}.png"
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap=plt.cm.Oranges)
    plt.title(f"Confusion Matrix on Test Set - {outer_test_well} ({attention_type})")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix: {cm_path}")

    # 保存 PR 曲线
    pr_path = f"pr_curve_test_outer_{outer_test_well}_{attention_type}.png"
    plot_precision_recall_curve(
        test_result["labels"],
        test_result["probs"],
        NUM_CLASSES,
        label_names,
        f"Precision-Recall Curve on Test Set - {outer_test_well} ({attention_type})",
        pr_path
    )
    print(f"Saved PR curve: {pr_path}")

    return {
        "test_acc": test_result["acc"],
        "test_loss": test_result["loss"],
        "report_text": report_text,
        "confusion_matrix": cm,
    }


# ==============================================================================
# Main: Two-level LOO
# ==============================================================================


def main():
    transform = get_transform()
    all_wells = get_all_wells(DATA_DIR)

    if len(all_wells) < 3:
        raise RuntimeError(
            "Two-level LOO needs at least 3 wells: one outer test + at least two outer-train wells."
        )

    print("="*90)
    print(f"Attention Ablation Study with Two-Level Leave-One-Well-Out")
    print(f"Attention Type: {ATTENTION}")
    print(f"Detected wells: {all_wells}")
    print(f"Device: {device}")
    print(f"Config: INNER_EPOCHS={INNER_EPOCHS}, LEARNING_RATE={LEARNING_RATE}")
    print(f"USE_WEIGHTED_SAMPLER={USE_WEIGHTED_SAMPLER}, USE_CLASS_WEIGHT_IN_LOSS={USE_CLASS_WEIGHT_IN_LOSS}")
    print("="*90)

    if USE_WEIGHTED_SAMPLER and USE_CLASS_WEIGHT_IN_LOSS:
        raise RuntimeError(
            "Both weighted sampler and weighted classification loss are enabled. "
            "Disable one to avoid double class reweighting."
        )

    outer_results = []

    for outer_test_well in all_wells:
        print("\n" + "=" * 90)
        print(f"Outer fold start: test well = {outer_test_well}")
        outer_train_wells = [w for w in all_wells if w != outer_test_well]
        print(f"Outer train wells: {outer_train_wells}")

        best_epoch, inner_curve = inner_loo_select_best_epoch(
            outer_train_wells=outer_train_wells,
            transform=transform,
            fixed_epochs=INNER_EPOCHS,
            attention_type=ATTENTION,
        )
        print(f"Selected best epoch for outer test={outer_test_well}: {best_epoch}")

        inner_curve_path = f"inner_curve_outer_test_{outer_test_well}_{ATTENTION}.csv"
        inner_curve.to_csv(inner_curve_path, index=False)
        print(f"Saved inner-LOO epoch curve: {inner_curve_path}")

        final_result = train_full_outer_and_test(
            outer_train_wells=outer_train_wells,
            outer_test_well=outer_test_well,
            transform=transform,
            best_epoch=best_epoch,
            attention_type=ATTENTION,
        )

        print(f"Outer test acc ({outer_test_well}): {final_result['test_acc']:.2f}%")
        print("Outer test classification report:")
        print(final_result["report_text"])
        print("Outer test confusion matrix:")
        print(final_result["confusion_matrix"])

        outer_results.append(
            {
                "outer_test_well": outer_test_well,
                "outer_train_wells": ",".join(outer_train_wells),
                "best_epoch": best_epoch,
                "test_acc": final_result["test_acc"],
                "test_loss": final_result["test_loss"],
            }
        )

    results_df = pd.DataFrame(outer_results)
    results_path = f"outer_loo_results_attention_{ATTENTION}.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 90)
    print("Two-level LOO complete.")
    print(results_df)
    print(f"Mean outer test acc: {results_df['test_acc'].mean():.2f}%")
    print(f"Saved summary to {results_path}")


if __name__ == "__main__":
    main()