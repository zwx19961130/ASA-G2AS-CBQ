# -*- coding: utf-8 -*-
# run_guided_attention_training.py

import os
import copy
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# ==============================================================================
# Global Config
# ==============================================================================

SEED = 42
DATA_DIR = "vdl_slices_20px"
BATCH_SIZE = 32
NUM_CLASSES = 3
LEARNING_RATE = 2e-5
IMAGE_SIZE = (500, 40)
INNER_EPOCHS = 50
LAMBDA_GUIDANCE = float(os.environ.get("LAMBDA", 5.0))
USE_ASA = True
USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHT_IN_LOSS = False
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 2.0
NUM_WORKERS = 4
# Optional: set to a list of expected wells to enforce full-dataset coverage.
EXPECTED_WELLS = None

label_map = {"Good": 0, "Midrate": 1, "Poor": 2}
label_names = ["Good", "Midrate", "Poor"]


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
# Model
# ==============================================================================


class AnisotropicSpatialAttention(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.horizontal_conv = nn.Conv1d(
            in_planes,
            in_planes,
            kernel_size=21,
            padding=10,
            groups=in_planes,
            bias=False,
        )
        self.vertical_conv = nn.Conv1d(
            in_planes,
            in_planes,
            kernel_size=3,
            padding=1,
            groups=in_planes,
            bias=False,
        )
        self.fusion_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_h_pooled = x.mean(dim=2)
        h_attn = self.horizontal_conv(x_h_pooled)
        h_attn = h_attn.unsqueeze(2).expand(-1, -1, x.shape[2], -1)

        x_v_pooled = x.mean(dim=3)
        v_attn = self.vertical_conv(x_v_pooled)
        v_attn = v_attn.unsqueeze(3).expand(-1, -1, -1, x.shape[3])

        fused_attn = h_attn + v_attn
        avg_out = torch.mean(fused_attn, dim=1, keepdim=True)
        max_out, _ = torch.max(fused_attn, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.fusion_conv(attention_map)
        return self.sigmoid(attention_map)


class LightweightVDLNet_PlacementAblation(nn.Module):
    def __init__(self, num_classes=3, apply_asa=True, asa_before_pool=True):
        super().__init__()
        self.apply_asa = apply_asa
        self.asa_before_pool = asa_before_pool

        self.block1_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1_pool = nn.MaxPool2d(2)

        self.block2_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2_pool = nn.MaxPool2d(2)

        self.block3_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block3_pool = nn.MaxPool2d(2)

        if self.apply_asa:
            self.asa1 = AnisotropicSpatialAttention(32)
            self.asa2 = AnisotropicSpatialAttention(64)
            self.asa3 = AnisotropicSpatialAttention(128)

        # Exposed for guidance loss (set each forward pass).
        self.latest_guidance_features = None
        self.latest_guidance_attention = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        self.latest_guidance_features = None
        self.latest_guidance_attention = None

        x = self.block1_conv(x)
        if self.apply_asa and self.asa_before_pool:
            attn1 = self.asa1(x)
            x = x + x * attn1.expand_as(x)
        x = self.block1_pool(x)
        if self.apply_asa and not self.asa_before_pool:
            attn1 = self.asa1(x)
            x = x + x * attn1.expand_as(x)

        x = self.block2_conv(x)
        if self.apply_asa and self.asa_before_pool:
            self.latest_guidance_features = x
            attn2 = self.asa2(x)
            self.latest_guidance_attention = attn2
            x = x + x * attn2.expand_as(x)
        x = self.block2_pool(x)
        if self.apply_asa and not self.asa_before_pool:
            self.latest_guidance_features = x
            attn2 = self.asa2(x)
            self.latest_guidance_attention = attn2
            x = x + x * attn2.expand_as(x)

        x = self.block3_conv(x)
        if self.apply_asa and self.asa_before_pool:
            attn3 = self.asa3(x)
            x = x + x * attn3.expand_as(x)
        x = self.block3_pool(x)
        if self.apply_asa and not self.asa_before_pool:
            attn3 = self.asa3(x)
            x = x + x * attn3.expand_as(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input_tensor, target_tensor):
        logp = F.log_softmax(input_tensor, dim=1)
        ce = F.nll_loss(logp, target_tensor, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


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
        sampler = WeightedRandomSampler(
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
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion_cls(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    acc = 100.0 * (np.array(all_preds) == np.array(all_labels)).mean() if all_labels else 0.0
    return {
        "loss": avg_loss,
        "acc": acc,
        "preds": all_preds,
        "labels": all_labels,
    }


def train_with_guidance(train_loader, val_loader, fixed_epochs):
    model = LightweightVDLNet_PlacementAblation(
        num_classes=NUM_CLASSES,
        apply_asa=USE_ASA,
        asa_before_pool=True,
    ).to(device)

    class_weights = (
        compute_class_weights(train_loader.dataset) if USE_CLASS_WEIGHT_IN_LOSS else None
    )
    if USE_FOCAL_LOSS:
        criterion_cls = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
    else:
        criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_guide = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    epoch_metrics = []
    best_val_state = None
    best_val_loss = float("inf")

    for epoch in range(fixed_epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{fixed_epochs}", leave=False)
        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            l_cls = criterion_cls(outputs, labels)

            if LAMBDA_GUIDANCE > 0:
                features_A = model.latest_guidance_features
                attention_S = model.latest_guidance_attention

                if features_A is None or attention_S is None:
                    raise RuntimeError(
                        "Guidance tensors are missing. Ensure apply_asa=True and forward pass sets guidance states."
                    )

                pred_classes = outputs.argmax(dim=1)
                target_scores = outputs.gather(1, pred_classes.unsqueeze(1)).squeeze()

                gradients = torch.autograd.grad(
                    outputs=target_scores.sum(),
                    inputs=features_A,
                    retain_graph=True,
                )[0]

                weights = gradients.mean(dim=(2, 3), keepdim=True)
                guidance_map_G = torch.sum(weights * features_A, dim=1, keepdim=True)
                guidance_map_G = F.relu(guidance_map_G)

                b_size = guidance_map_G.shape[0]
                g_flat = guidance_map_G.view(b_size, -1)
                g_min = g_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
                g_max = g_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
                guidance_map_G = (guidance_map_G - g_min) / (g_max - g_min + 1e-8)

                l_guide = criterion_guide(attention_S, guidance_map_G.detach())
            else:
                l_guide = torch.tensor(0.0, device=device)

            total_loss = l_cls + LAMBDA_GUIDANCE * l_guide

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            loop.set_postfix(train_loss=f"{running_loss / max(1, len(train_loader)):.4f}")

        val_result = evaluate(model, val_loader, criterion_cls)
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "val_loss": val_result["loss"],
                "val_acc": val_result["acc"],
            }
        )

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            best_val_state = copy.deepcopy(model.state_dict())

    return {
        "epoch_metrics": epoch_metrics,
        "best_val_state": best_val_state,
    }


def inner_loo_select_best_epoch(outer_train_wells, transform, fixed_epochs):
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
        fold_result = train_with_guidance(train_loader, val_loader, fixed_epochs=fixed_epochs)
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


def train_full_outer_and_test(outer_train_wells, outer_test_well, transform, best_epoch):
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
        apply_asa=USE_ASA,
        asa_before_pool=True,
    ).to(device)

    class_weights = compute_class_weights(train_ds) if USE_CLASS_WEIGHT_IN_LOSS else None
    if USE_FOCAL_LOSS:
        criterion_cls = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
    else:
        criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_guide = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    print(f"  Final outer training for {best_epoch} epochs on wells={outer_train_wells}")
    for epoch in range(best_epoch):
        model.train()
        loop = tqdm(train_loader, desc=f"Final train epoch {epoch + 1}/{best_epoch}", leave=False)

        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            l_cls = criterion_cls(outputs, labels)

            if LAMBDA_GUIDANCE > 0:
                features_A = model.latest_guidance_features
                attention_S = model.latest_guidance_attention

                if features_A is None or attention_S is None:
                    raise RuntimeError(
                        "Guidance tensors are missing. Ensure apply_asa=True and forward pass sets guidance states."
                    )

                pred_classes = outputs.argmax(dim=1)
                target_scores = outputs.gather(1, pred_classes.unsqueeze(1)).squeeze()

                gradients = torch.autograd.grad(
                    outputs=target_scores.sum(),
                    inputs=features_A,
                    retain_graph=True,
                )[0]

                weights = gradients.mean(dim=(2, 3), keepdim=True)
                guidance_map_G = torch.sum(weights * features_A, dim=1, keepdim=True)
                guidance_map_G = F.relu(guidance_map_G)

                b_size = guidance_map_G.shape[0]
                g_flat = guidance_map_G.view(b_size, -1)
                g_min = g_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
                g_max = g_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
                guidance_map_G = (guidance_map_G - g_min) / (g_max - g_min + 1e-8)

                l_guide = criterion_guide(attention_S, guidance_map_G.detach())
            else:
                l_guide = torch.tensor(0.0, device=device)

            total_loss = l_cls + LAMBDA_GUIDANCE * l_guide

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # 保存模型
    model_path = f"guided_model_outer_{outer_test_well}_ASA{USE_ASA}_lambda{LAMBDA_GUIDANCE}.pt"
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

    print(f"Detected wells: {all_wells}")
    print(f"Evaluation scope is limited to wells under DATA_DIR='{DATA_DIR}'.")

    if EXPECTED_WELLS is not None:
        expected_sorted = sorted(EXPECTED_WELLS)
        if sorted(all_wells) != expected_sorted:
            raise RuntimeError(
                f"Detected wells {all_wells} do not match EXPECTED_WELLS {expected_sorted}. "
                "This run would only support subset-level conclusions."
            )

    print(f"Device: {device}")
    print(
        f"Config: INNER_EPOCHS={INNER_EPOCHS}, LAMBDA_GUIDANCE={LAMBDA_GUIDANCE}, "
        f"USE_WEIGHTED_SAMPLER={USE_WEIGHTED_SAMPLER}, "
        f"USE_CLASS_WEIGHT_IN_LOSS={USE_CLASS_WEIGHT_IN_LOSS}, USE_FOCAL_LOSS={USE_FOCAL_LOSS}"
    )

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
        )
        print(f"Selected best epoch for outer test={outer_test_well}: {best_epoch}")

        inner_curve_path = f"inner_curve_outer_test_{outer_test_well}.csv"
        inner_curve.to_csv(inner_curve_path, index=False)
        print(f"Saved inner-LOO epoch curve: {inner_curve_path}")

        final_result = train_full_outer_and_test(
            outer_train_wells=outer_train_wells,
            outer_test_well=outer_test_well,
            transform=transform,
            best_epoch=best_epoch,
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
    results_df.to_csv("outer_loo_results.csv", index=False)

    print("\n" + "=" * 90)
    print("Two-level LOO complete.")
    print(results_df)
    print(f"Mean outer test acc: {results_df['test_acc'].mean():.2f}%")
    print("Saved summary to outer_loo_results.csv")


if __name__ == "__main__":
    main()
