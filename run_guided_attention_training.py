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
import time

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
LAMBDA_GUIDANCE = float(os.environ.get("LAMBDA", 0.1))  # 建议范围: 0.001~0.1
USE_ASA = True
USE_WEIGHTED_SAMPLER = True
EMA_DECAY = 0.99  # EMA teacher 衰减率
WARMUP_EPOCHS = 3  # warmup 阶段，前 N 个 epoch 不启用 guidance，只训练分类
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

    def forward(self, x, enable_asa=None, return_guidance=False):
        """
        enable_asa:
            True/False 强制开关 ASA；None 表示使用 self.apply_asa
        return_guidance:
            True 时返回 (logits, feat2, attn2)；否则只返回 logits
        """
        if enable_asa is None:
            enable_asa = self.apply_asa

        # 只有当 apply_asa=True 时才真正启用 ASA
        enable_asa = enable_asa and self.apply_asa

        feat2 = None
        attn2 = None

        x = self.block1_conv(x)
        if enable_asa and self.asa_before_pool:
            a1 = self.asa1(x)
            x = x + x * a1.expand_as(x)
        x = self.block1_pool(x)
        if enable_asa and (not self.asa_before_pool):
            a1 = self.asa1(x)
            x = x + x * a1.expand_as(x)

        x = self.block2_conv(x)

        # --- 关键：无论 enable_asa 是否开启，都在"ASA2应当作用的位置"取 feat2 ---
        if self.asa_before_pool:
            feat2 = x
            if enable_asa:
                attn2 = self.asa2(x)
                x = x + x * attn2.expand_as(x)
        x = self.block2_pool(x)
        if not self.asa_before_pool:
            feat2 = x
            if enable_asa:
                attn2 = self.asa2(x)
                x = x + x * attn2.expand_as(x)

        x = self.block3_conv(x)
        if enable_asa and self.asa_before_pool:
            a3 = self.asa3(x)
            x = x + x * a3.expand_as(x)
        x = self.block3_pool(x)
        if enable_asa and (not self.asa_before_pool):
            a3 = self.asa3(x)
            x = x + x * a3.expand_as(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        if return_guidance:
            return logits, feat2, attn2
        return logits


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


def build_ema_teacher(student_model):
    """构建 EMA teacher 模型（student 的副本，冻结参数）"""
    teacher_model = copy.deepcopy(student_model)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    return teacher_model


def update_ema_teacher(student_model, teacher_model, ema_decay):
    """更新 EMA teacher 参数和 BN buffers: θ_teacher = α * θ_teacher + (1-α) * θ_student"""
    with torch.no_grad():
        # 更新参数 (weights & biases)
        for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
            t_param.data.mul_(ema_decay).add_(s_param.data, alpha=1.0 - ema_decay)

        # 更新 BN buffers (running_mean, running_var)
        # 注意：只对浮点型 buffer 做 EMA，非浮点 buffer（如 num_batches_tracked）直接 copy
        for t_buffer, s_buffer in zip(teacher_model.buffers(), student_model.buffers()):
            if t_buffer.dtype.is_floating_point:
                t_buffer.data.mul_(ema_decay).add_(s_buffer.data, alpha=1.0 - ema_decay)
            else:
                t_buffer.data.copy_(s_buffer.data)


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


def count_parameters(model):
    """统计模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def training_step(student_model, teacher_model, inputs, labels, optimizer, criterion_cls, lambda_guidance):
    """
    统一的训练步骤：student forward + EMA teacher forward for GGAS

    返回: (total_loss, l_cls, l_guide)
    """
    inputs = inputs.to(device)
    labels = labels.to(device)

    # === student 分支：根据 USE_ASA 决定是否开启 ASA ===
    outputs, feat2_s, attn2_s = student_model(
        inputs,
        enable_asa=USE_ASA,
        return_guidance=True
    )
    l_cls = criterion_cls(outputs, labels)

    # === teacher 分支：使用 EMA teacher 生成 Grad-CAM 指导图 G ===
    # 只有当 LAMBDA_GUIDANCE > 0 且 USE_ASA=True 时才计算 guidance
    l_guide = torch.tensor(0.0, device=device)
    conf_mask = torch.tensor(0.0, device=device)

    if lambda_guidance > 0 and USE_ASA:
        if attn2_s is None:
            raise RuntimeError("attn2_s is None. USE_ASA=True 时应当产生 ASA2 注意力。")

        # 使用 EMA teacher（关闭 ASA）生成 Grad-CAM 目标
        # 关键修复：创建 requires_grad=True 的输入副本，确保 feat2_t 能够正确进入可求导图
        inputs_t = inputs.detach().requires_grad_(True)
        teacher_model.eval()
        logits_t, feat2_t, _ = teacher_model(
            inputs_t,
            enable_asa=False,
            return_guidance=True
        )

        # ===== 改进 1: 使用 teacher 预测的类别而不是真实标签 =====
        # 这样 CAM 来自 teacher 当前最 confident 的类别，避免早期被错误标签结构干扰
        pred_classes = logits_t.argmax(dim=1)
        target_scores_t = logits_t.gather(1, pred_classes.unsqueeze(1)).squeeze(1)

        # ===== 可选: 置信度过滤 =====
        # 只对高置信度样本加强监督，低置信度样本不强监督
        conf = torch.softmax(logits_t, dim=1).max(dim=1)[0]
        conf_mask = (conf > 0.5).float()  # 置信度 > 50% 才计算 guide loss

        # 计算 Grad-CAM（不对 teacher forward 包 no_grad，因为需要 grad）
        gradients = torch.autograd.grad(
            outputs=target_scores_t.sum(),
            inputs=feat2_t,
            retain_graph=False,
            create_graph=False,
        )[0]

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        guidance_map_G = torch.sum(weights * feat2_t, dim=1, keepdim=True)
        guidance_map_G = F.relu(guidance_map_G)

        # 改进的归一化：除以 mean（保持相对结构），加 clamp 防止极端值
        g_mean = guidance_map_G.mean(dim=(2, 3), keepdim=True)
        guidance_map_G = guidance_map_G / (g_mean + 1e-6)
        guidance_map_G = guidance_map_G.clamp(max=5.0)

        # 尺寸安全检查
        assert guidance_map_G.shape == attn2_s.shape, \
            f"Shape mismatch: guidance_map_G {guidance_map_G.shape} vs attn2_s {attn2_s.shape}"

        # ===== 改进 2: 使用 KL divergence 匹配空间分布而不是 MSE 匹配绝对值 =====
        # 将 attention 和 guidance map 转为概率分布，然后计算 KL divergence
        b = attn2_s.shape[0]

        # flatten spatial dimensions
        attn_flat = attn2_s.view(b, -1)
        g_flat = guidance_map_G.view(b, -1)

        # 转成概率分布 (softmax over spatial dimensions)
        attn_prob = F.softmax(attn_flat, dim=1)
        g_prob = F.softmax(g_flat, dim=1)

        # KL divergence: KL(student || teacher) - student 分布去逼近 teacher 分布
        # 重要：detach 的是 teacher 目标 g_prob，保留 student attn_prob 以获得梯度
        # 方向：log(attn_prob) 是 input，g_prob 是 target
        guide_raw = F.kl_div(
            torch.log(attn_prob + 1e-8),
            g_prob.detach(),
            reduction="none"
        ).sum(dim=1)

        # 使用 conf_mask 过滤低置信度样本，并用 sum/denom 避免被 0 稀释
        l_guide = (guide_raw * conf_mask).sum() / (conf_mask.sum() + 1e-6)

    total_loss = l_cls + lambda_guidance * l_guide

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 更新 EMA teacher（无条件更新，确保 warmup 期间 teacher 也会跟随 student 更新）
    update_ema_teacher(student_model, teacher_model, EMA_DECAY)

    return total_loss, l_cls.detach(), l_guide.detach(), conf_mask.mean().detach()


def train_with_guidance(train_loader, val_loader, fixed_epochs):
    model = LightweightVDLNet_PlacementAblation(
        num_classes=NUM_CLASSES,
        apply_asa=USE_ASA,
        asa_before_pool=True,
    ).to(device)

    # 创建 EMA teacher 模型
    teacher_model = build_ema_teacher(model)

    class_weights = (
        compute_class_weights(train_loader.dataset) if USE_CLASS_WEIGHT_IN_LOSS else None
    )
    if USE_FOCAL_LOSS:
        criterion_cls = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
    else:
        criterion_cls = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    epoch_metrics = []
    best_val_state = None
    best_val_loss = float("inf")

    for epoch in range(fixed_epochs):
        model.train()
        running_loss = 0.0

        # 计时与显存统计
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        # 累加变量，用于统计 GGAS 信息
        sum_cls = 0.0
        sum_guide = 0.0
        sum_conf = 0.0
        num_batches = 0

        # warmup: 前 WARMUP_EPOCHS 个 epoch 不启用 guidance，只训练分类
        if epoch < WARMUP_EPOCHS:
            effective_lambda = 0.0
            guidance_status = "warmup"
        else:
            effective_lambda = LAMBDA_GUIDANCE
            guidance_status = "active"

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{fixed_epochs} (guidance: {guidance_status})", leave=False)
        for inputs, labels in loop:
            total_loss, l_cls, l_guide, conf_mean = training_step(
                student_model=model,
                teacher_model=teacher_model,
                inputs=inputs,
                labels=labels,
                optimizer=optimizer,
                criterion_cls=criterion_cls,
                lambda_guidance=effective_lambda
            )

            running_loss += total_loss.item()
            # 累计 GGAS 统计信息
            sum_cls += l_cls.item()
            sum_guide += l_guide.item()
            sum_conf += conf_mean.item()
            num_batches += 1

            loop.set_postfix(train_loss=f"{running_loss / max(1, num_batches):.4f}")

        # epoch 结束时打印 GGAS 统计信息
        if num_batches > 0:
            mean_cls = sum_cls / num_batches
            mean_guide = sum_guide / num_batches
            mean_lambda_guide = effective_lambda * mean_guide
            mean_conf = sum_conf / num_batches

            print(
                f"[GGAS stats] "
                f"mean_l_cls={mean_cls:.4f} "
                f"mean_l_guide={mean_guide:.4f} "
                f"lambda*l_guide={mean_lambda_guide:.4f} "
                f"conf_mask_mean={mean_conf:.4f}"
            )

        # 计算训练时间和显存使用
        epoch_time = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        val_result = evaluate(model, val_loader, criterion_cls)
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "val_loss": val_result["loss"],
                "val_acc": val_result["acc"],
                "time_per_epoch_sec": epoch_time,
                "gpu_mem_gb": peak_mem
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

    return best_epoch, merged, inner_histories


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

    # 创建 EMA teacher 模型
    teacher_model = build_ema_teacher(model)

    class_weights = compute_class_weights(train_ds) if USE_CLASS_WEIGHT_IN_LOSS else None
    if USE_FOCAL_LOSS:
        criterion_cls = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
    else:
        criterion_cls = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    print(f"  Final outer training for {best_epoch} epochs on wells={outer_train_wells}")
    for epoch in range(best_epoch):
        model.train()

        # 累加变量，用于统计 GGAS 信息
        sum_cls = 0.0
        sum_guide = 0.0
        sum_conf = 0.0
        num_batches = 0

        # warmup: 前 WARMUP_EPOCHS 个 epoch 不启用 guidance，只训练分类
        if epoch < WARMUP_EPOCHS:
            effective_lambda = 0.0
            guidance_status = "warmup"
        else:
            effective_lambda = LAMBDA_GUIDANCE
            guidance_status = "active"

        loop = tqdm(train_loader, desc=f"Final train epoch {epoch + 1}/{best_epoch} (guidance: {guidance_status})", leave=False)

        for inputs, labels in loop:
            total_loss, l_cls, l_guide, conf_mean = training_step(
                student_model=model,
                teacher_model=teacher_model,
                inputs=inputs,
                labels=labels,
                optimizer=optimizer,
                criterion_cls=criterion_cls,
                lambda_guidance=effective_lambda
            )

            # 累计 GGAS 统计信息
            sum_cls += l_cls.item()
            sum_guide += l_guide.item()
            sum_conf += conf_mean.item()
            num_batches += 1

        # epoch 结束时打印 GGAS 统计信息
        if num_batches > 0:
            mean_cls = sum_cls / num_batches
            mean_guide = sum_guide / num_batches
            mean_lambda_guide = effective_lambda * mean_guide
            mean_conf = sum_conf / num_batches

            print(
                f"[GGAS stats] "
                f"mean_l_cls={mean_cls:.4f} "
                f"mean_l_guide={mean_guide:.4f} "
                f"lambda*l_guide={mean_lambda_guide:.4f} "
                f"conf_mask_mean={mean_conf:.4f}"
            )

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

    # 打印模型参数量
    model = LightweightVDLNet_PlacementAblation(
        num_classes=NUM_CLASSES,
        apply_asa=USE_ASA,
        asa_before_pool=True,
    ).to(device)
    params = count_parameters(model)
    print(f"Model Parameters: {params / 1e6:.2f} M")

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

        best_epoch, inner_curve, inner_histories = inner_loo_select_best_epoch(
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

        # 统计训练时间和显存使用（使用 inner_histories 而不是 inner_curve）
        all_times = []
        all_mems = []

        for hist in inner_histories:
            all_times.extend(hist["time_per_epoch_sec"])
            all_mems.extend(hist["gpu_mem_gb"])

        avg_epoch_time = np.mean(all_times)
        avg_gpu_mem = np.max(all_mems)

        outer_results.append(
            {
                "outer_test_well": outer_test_well,
                "outer_train_wells": ",".join(outer_train_wells),
                "best_epoch": best_epoch,
                "test_acc": final_result["test_acc"],
                "test_loss": final_result["test_loss"],
                "train_time_epoch_sec": avg_epoch_time,
                "gpu_memory_gb": avg_gpu_mem,
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
