# -*- coding: utf-8 -*-
# run_guided_attention_training.py

import os
import copy
import random

# ===== Normalization control =====
NORM_TYPE = os.environ.get("NORM_TYPE", "BN")  
# options: BN | IN | NONE

def Norm2d(ch):
    if NORM_TYPE == "BN":
        return nn.BatchNorm2d(ch)
    elif NORM_TYPE == "IN":
        return nn.InstanceNorm2d(ch, affine=True)
    elif NORM_TYPE == "NONE":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported NORM_TYPE '{NORM_TYPE}', must be BN, IN, or NONE")


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

SEED = int(os.environ.get("SEED", "42"))
VERTICAL_RESOLUTION = int(os.environ.get("VERTICAL_RESOLUTION", "250"))
DATA_DIR = "vdl_slices_20px"
BATCH_SIZE = 32
NUM_CLASSES = 3
LEARNING_RATE = 2e-5
IMAGE_SIZE = (VERTICAL_RESOLUTION, 40)
INNER_EPOCHS = 50
LAMBDA_GUIDANCE = float(os.environ.get("LAMBDA", 0.0))  # Recommended range: 0.001~0.1
ATTENTION = os.environ.get("ATTENTION", "ASA")  # options: "ASA", "SE", "ECA", "CBAM", "None"
if ATTENTION == "None":
    ATTENTION = None

# ===== ASA ablation switches =====
ASA_PLACEMENT = os.environ.get("ASA_PLACEMENT", "after")  
# options: before | after
if ASA_PLACEMENT not in ("before", "after"):
    raise ValueError(f"Invalid ASA_PLACEMENT '{ASA_PLACEMENT}'; must be 'before' or 'after'.")

ASA_COMPONENT = os.environ.get("ASA_COMPONENT", "full")  
# options: full | horizontal | vertical

ASA_FUSION = os.environ.get("ASA_FUSION", "gate")  
# options: sum | weighted | gate
if ASA_FUSION not in ("sum", "weighted", "gate"):
    raise ValueError(f"Invalid ASA_FUSION '{ASA_FUSION}'; must be 'sum', 'weighted' or 'gate'.")
if ASA_COMPONENT not in ("full", "horizontal", "vertical"):
    raise ValueError(f"Invalid ASA_COMPONENT '{ASA_COMPONENT}'; must be 'full', 'horizontal' or 'vertical'.")

ASA_HPOOL = os.environ.get("ASA_HPOOL", "conv")  
# "mean" = current design (mean over H)
# "conv" = keep depth position (no mean pooling)
if ASA_HPOOL not in ("mean", "conv"):
    raise ValueError(f"Invalid ASA_HPOOL '{ASA_HPOOL}'; must be 'mean' or 'conv'.")

# switches for individual attention placements (layer1, layer2, layer3)
ATT_L1 = bool(int(os.environ.get("ATT_L1", "0")))
ATT_L2 = bool(int(os.environ.get("ATT_L2", "1")))
ATT_L3 = bool(int(os.environ.get("ATT_L3", "1")))  # set to 0/1 if you prefer environment control

# data augmentation flags
USE_HFLIP = bool(int(os.environ.get("USE_HFLIP", "0")))
USE_TIMESHIFT = bool(int(os.environ.get("USE_TIMESHIFT", "0")))
TIMESHIFT_MAX = int(os.environ.get("TIMESHIFT_MAX", "2"))  # Start with 2 recommended


USE_WEIGHTED_SAMPLER = True
EMA_DECAY = 0.99  # EMA teacher decay factor
WARMUP_EPOCHS = 2  # warmup stage: first N epochs disable guidance, train classification only
USE_CLASS_WEIGHT_IN_LOSS = False
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 2.0
# for reproducibility you can set NUM_WORKERS=0; otherwise workers are seeded
NUM_WORKERS = 0  # use 0 when you want exact deterministic loader behaviour
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # make CUDA/cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


# limit CPU parallelism for determinism
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

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

                for fname in sorted(os.listdir(class_dir)):
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
    def __init__(self, in_planes, component="full", hpool="mean"):
        super().__init__()
        self.component = component
        self.fusion_mode = ASA_FUSION
        self.hpool = hpool

        if self.hpool == "mean":
            self.horizontal_conv = nn.Conv1d(
                in_planes,
                in_planes,
                kernel_size=21,
                padding=10,
                groups=in_planes,
                bias=False,
            )
            self.horizontal_conv2d = None
        else:
            self.horizontal_conv = None
            self.horizontal_conv2d = nn.Conv2d(
                in_planes,
                in_planes,
                kernel_size=(1, 21),
                padding=(0, 10),
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

        # weighted fusion
        if self.fusion_mode == "weighted":
            # parameter unconstrained; will be squashed in forward
            self.raw_alpha = nn.Parameter(torch.tensor(0.0))

        # gating fusion
        if self.fusion_mode == "gate":
            self.gate = nn.Sequential(
                nn.Conv2d(in_planes, 1, kernel_size=1),
                nn.Sigmoid()
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.hpool == "mean":
            h_attn = x.mean(dim=2)
            h_attn = self.horizontal_conv(h_attn)
            h_attn = h_attn.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
        else:
            h_attn = self.horizontal_conv2d(x)

        x_v_pooled = x.mean(dim=3)
        v_attn = self.vertical_conv(x_v_pooled)
        v_attn = v_attn.unsqueeze(3).expand(-1, -1, -1, x.shape[3])

        if self.component == "horizontal":
            fused_attn = h_attn
        elif self.component == "vertical":
            fused_attn = v_attn
        else:
            if self.fusion_mode == "sum":
                fused_attn = h_attn + v_attn
            elif self.fusion_mode == "weighted":
                alpha = torch.sigmoid(self.raw_alpha)
                fused_attn = alpha * h_attn + (1 - alpha) * v_attn
            elif self.fusion_mode == "gate":
                g = self.gate(x)
                fused_attn = g * h_attn + (1 - g) * v_attn
        avg_out = torch.mean(fused_attn, dim=1, keepdim=True)
        max_out, _ = torch.max(fused_attn, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention_logits = self.fusion_conv(attention_map)
        attention_map = torch.sigmoid(attention_logits)
        return attention_logits, attention_map


class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y.expand_as(x)


class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return y.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels)
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        b, c, h, w = x.shape

        avg = torch.mean(x, dim=(2, 3))
        mx = torch.amax(x, dim=(2, 3))
        ca = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(mx)).view(b, c, 1, 1)
        x_ca = x * ca

        avg = torch.mean(x_ca, dim=1, keepdim=True)
        mx, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))
        gate = ca * sa
        return gate.expand_as(x)


class GateIdentity(nn.Module):
    def forward(self, x):
        return torch.ones_like(x)


def build_attention(att_type, channels):
    if att_type == "ASA":
        return AnisotropicSpatialAttention(
            channels,
            component=ASA_COMPONENT,
            hpool=ASA_HPOOL,
        )
    if att_type == "SE":
        return SEBlock(channels)
    if att_type == "ECA":
        return ECABlock(channels)
    if att_type == "CBAM":
        return CBAM(channels)
    return GateIdentity()


class LightweightVDLNet_PlacementAblation(nn.Module):
    def __init__(self, num_classes=3, attention="ASA"):
        super().__init__()
        self.attention = attention
        self.asa_before_pool = (ASA_PLACEMENT == "before")

        self.block1_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            Norm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            Norm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1_pool = nn.MaxPool2d(2)

        self.block2_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            Norm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            Norm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2_pool = nn.MaxPool2d(2)

        self.block3_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            Norm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            Norm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block3_pool = nn.MaxPool2d(2)

        self.att1 = build_attention(self.attention, 32)
        self.att2 = build_attention(self.attention, 64)
        self.att3 = build_attention(self.attention, 128)



        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, enable_attention=None, return_guidance=False):
        """
        enable_attention:
            True/False force attention on/off; None means use self.attention setting
        return_guidance:
            If True, return (logits, feat2, attn2_logits, attn2_map); otherwise return logits only
        """
        if enable_attention is None:
            enable_attention = self.attention is not None

        enable_attention = enable_attention and (self.attention is not None)

        feat2 = None
        attn2 = None
        attn2_logits = None
        attn2_map = None

        x = self.block1_conv(x)
        # layer‑1 attention controlled by ATT_L1
        if enable_attention and ATT_L1 and self.asa_before_pool:
            if self.attention == "ASA":
                a1_logits, a1 = self.att1(x)
                x = x * a1.expand_as(x)
            else:
                g1 = self.att1(x)
                x = x * g1
        x = self.block1_pool(x)
        if enable_attention and ATT_L1 and (not self.asa_before_pool):
            if self.attention == "ASA":
                a1_logits, a1 = self.att1(x)
                x = x * a1.expand_as(x)
            else:
                g1 = self.att1(x)
                x = x * g1

        x = self.block2_conv(x)

        # --- Key: regardless of enable_attention, capture feat2 at the point where attention2 should apply ---
        if self.asa_before_pool:
            feat2 = x
            if enable_attention and ATT_L2:
                if self.attention == "ASA":
                    a2_logits, a2 = self.att2(x)
                    attn2_logits = a2_logits
                    attn2 = a2
                    attn2_map = a2
                    x = x * a2.expand_as(x)
                else:
                    g2 = self.att2(x)
                    x = x * g2
        x = self.block2_pool(x)
        if not self.asa_before_pool:
            feat2 = x
            if enable_attention and ATT_L2:
                if self.attention == "ASA":
                    a2_logits, a2 = self.att2(x)
                    attn2_logits = a2_logits
                    attn2 = a2
                    attn2_map = a2
                    x = x * a2.expand_as(x)
                else:
                    g2 = self.att2(x)
                    x = x * g2

        x = self.block3_conv(x)
        if enable_attention and ATT_L3 and self.asa_before_pool:
            if self.attention == "ASA":
                a3_logits, a3 = self.att3(x)
                x = x * a3.expand_as(x)
            else:
                g3 = self.att3(x)
                x = x * g3
        x = self.block3_pool(x)
        if enable_attention and ATT_L3 and (not self.asa_before_pool):
            if self.attention == "ASA":
                a3_logits, a3 = self.att3(x)
                x = x * a3.expand_as(x)
            else:
                g3 = self.att3(x)
                x = x * g3

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        if return_guidance:
            return logits, feat2, attn2_logits, attn2_map
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


class TimeShift:
    def __init__(self, max_shift=2, fill=0):
        self.max_shift = max_shift
        self.fill = fill

    def __call__(self, img):
        arr = np.array(img)
        shift = random.randint(-self.max_shift, self.max_shift)

        if shift == 0:
            return Image.fromarray(arr)

        out = np.full_like(arr, self.fill)

        # horizontal shift along the width axis
        if shift > 0:
            out[:, shift:] = arr[:, :-shift]
        else:
            out[:, :shift] = arr[:, -shift:]

        return Image.fromarray(out)


def get_transform(is_train=False):
    tfms = [transforms.Resize(IMAGE_SIZE)]

    if is_train:
        if USE_HFLIP:
            tfms.append(transforms.RandomHorizontalFlip(p=0.5))
        if USE_TIMESHIFT:
            tfms.append(TimeShift(max_shift=TIMESHIFT_MAX, fill=0))

    tfms.append(transforms.ToTensor())
    return transforms.Compose(tfms)


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_all_wells(root_dir):
    wells = sorted([w for w in os.listdir(root_dir) if is_valid_well_dir(root_dir, w)])
    return wells


def build_loader(dataset, is_train):
    # use a consistent generator for every loader call
    g = torch.Generator()
    g.manual_seed(SEED)

    if is_train and USE_WEIGHTED_SAMPLER:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        classes, counts = np.unique(labels, return_counts=True)
        class_weights_for_sampler = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
        sample_weights = np.array([class_weights_for_sampler[l] for l in labels])

        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True,
            generator=g,
        )
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
        )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=is_train,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )


def build_ema_teacher(student_model):
    """Build EMA teacher model (copy of student, with frozen parameters)."""
    teacher_model = copy.deepcopy(student_model)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    return teacher_model


def update_ema_teacher(student_model, teacher_model, ema_decay):
    """Update EMA teacher parameters and BN buffers: θ_teacher = α * θ_teacher + (1-α) * θ_student"""
    with torch.no_grad():
        # update parameters (weights & biases)
        for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
            t_param.data.mul_(ema_decay).add_(s_param.data, alpha=1.0 - ema_decay)

        # update BN buffers (running_mean, running_var)
        # Note: apply EMA only to floating-point buffers, copy non-floating buffers (e.g., num_batches_tracked)
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


def compute_teacher_gradcam_map(teacher_model, inputs):
    inputs_t = inputs.detach().requires_grad_(True)
    teacher_model.eval()
    logits_t, feat2_t, attn2_logits_t, _ = teacher_model(
        inputs_t,
        enable_attention=False,
        return_guidance=True,
    )

    pred_classes = logits_t.argmax(dim=1)
    target_scores_t = logits_t.gather(1, pred_classes.unsqueeze(1)).squeeze(1)
    conf = torch.softmax(logits_t, dim=1).max(dim=1)[0]
    conf_mask = (conf > 0.5).float()

    gradients = torch.autograd.grad(
        outputs=target_scores_t.sum(),
        inputs=feat2_t,
        retain_graph=False,
        create_graph=False,
    )[0]

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    guidance_map = torch.sum(weights * feat2_t, dim=1, keepdim=True)
    guidance_map = F.relu(guidance_map)

    g_mean = guidance_map.mean(dim=(2, 3), keepdim=True)
    guidance_map = guidance_map / (g_mean + 1e-6)
    guidance_map = guidance_map.clamp(max=5.0)

    return {
        "guidance_map": guidance_map,
        "conf_mask": conf_mask,
    }


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


def build_soft_time_masks(width, early=(0.0, 0.3), late=(0.5, 0.9)):
    early_mask = torch.zeros(width)
    late_mask = torch.zeros(width)

    for j in range(width):
        l = j / width
        r = (j + 1) / width
        overlap_early = max(0.0, min(r, early[1]) - max(l, early[0]))
        overlap_late = max(0.0, min(r, late[1]) - max(l, late[0]))
        early_mask[j] = overlap_early * width
        late_mask[j] = overlap_late * width

    return early_mask, late_mask


def attention_to_temporal_profile(attn_map):
    total = attn_map.sum(dim=(2, 3), keepdim=True)
    normalized = attn_map / (total + 1e-6)
    q = normalized.sum(dim=2)
    return q.squeeze(1)


def compute_entropy_from_attention(attn_map):
    normalized = attn_map / (attn_map.sum(dim=(2, 3), keepdim=True) + 1e-6)
    entropy = -(normalized * torch.log(normalized + 1e-8)).sum(dim=(2, 3))
    return entropy


def compute_physics_scores_from_profile(q):
    width = q.shape[1]
    early_mask, late_mask = build_soft_time_masks(width)
    early_mask = early_mask.to(q.device)
    late_mask = late_mask.to(q.device)

    early_mass = (q * early_mask).sum(dim=1)
    late_mass = (q * late_mask).sum(dim=1)
    union = early_mass + late_mass + 1e-6
    balance = 1.0 - torch.abs(early_mass - late_mass) / (union + 1e-6)
    midrate_score = union * balance

    background_mask = torch.clamp(1.0 - early_mask - late_mask, min=0.0)
    background_mass = (q * background_mask).sum(dim=1)

    return {
        "early_mass": early_mass,
        "late_mass": late_mass,
        "midrate_score": midrate_score,
        "background_mass": background_mass,
    }


def compute_map_corr_and_iou(attn_map, gradcam_map, top_ratio=0.2):
    batch = attn_map.shape[0]
    attn_flat = attn_map.view(batch, -1)
    grad_flat = gradcam_map.view(batch, -1)

    attn_centered = attn_flat - attn_flat.mean(dim=1, keepdim=True)
    grad_centered = grad_flat - grad_flat.mean(dim=1, keepdim=True)

    numerator = (attn_centered * grad_centered).sum(dim=1)
    denominator = torch.sqrt(
        (attn_centered ** 2).sum(dim=1) * (grad_centered ** 2).sum(dim=1)
    ) + 1e-6
    pearson = numerator / denominator

    k = max(1, int(attn_flat.shape[1] * top_ratio))
    attn_thresh = torch.topk(attn_flat, k, dim=1)[0][:, -1]
    grad_thresh = torch.topk(grad_flat, k, dim=1)[0][:, -1]

    attn_peak = attn_flat >= attn_thresh.unsqueeze(1)
    grad_peak = grad_flat >= grad_thresh.unsqueeze(1)
    intersection = (attn_peak & grad_peak).sum(dim=1).float()
    union = (attn_peak | grad_peak).sum(dim=1).float()
    iou = intersection / (union + 1e-6)

    return pearson, iou


def evaluate_explainability(model, teacher_model, loader, correct_only=True):
    model.eval()
    teacher_model.eval()

    sample_rows = []
    early_scores, late_scores, mid_scores, background_scores = [], [], [], []
    entropies, pearsons, ious = [], [], []
    poor_early = []
    good_late = []
    midrate_vals = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs, feat2, attn2_logits, attn2_map = model(
            inputs,
            enable_attention=(ATTENTION is not None),
            return_guidance=True,
        )

        if attn2_map is None:
            continue

        preds = outputs.argmax(dim=1)
        mask = preds == labels if correct_only else torch.ones_like(preds, dtype=torch.bool)
        if mask.sum() == 0:
            continue

        selected_preds = preds[mask]
        selected_labels = labels[mask]

        gradcam_info = compute_teacher_gradcam_map(teacher_model, inputs)
        gradcam_map = gradcam_info["guidance_map"][mask]

        attn2_map = attn2_map[mask]
        q = attention_to_temporal_profile(attn2_map)
        physics_scores = compute_physics_scores_from_profile(q)

        entropy_vals = compute_entropy_from_attention(attn2_map)
        pearson_vals, iou_vals = compute_map_corr_and_iou(attn2_map, gradcam_map)

        for idx in range(q.shape[0]):
            early = physics_scores["early_mass"][idx]
            late = physics_scores["late_mass"][idx]
            mid = physics_scores["midrate_score"][idx]
            background = physics_scores["background_mass"][idx]
            ent = entropy_vals[idx]
            pear = pearson_vals[idx]
            iou = iou_vals[idx]

            sample_rows.append(
                {
                    "pred": int(selected_preds[idx].item()),
                    "label": int(selected_labels[idx].item()),
                    "early_mass": float(early.item()),
                    "late_mass": float(late.item()),
                    "midrate_score": float(mid.item()),
                    "background_mass": float(background.item()),
                    "entropy": float(ent.item()),
                    "pearson_corr": float(pear.item()),
                    "iou_top20": float(iou.item()),
                }
            )

        early_scores.extend(physics_scores["early_mass"].detach().cpu().tolist())
        late_scores.extend(physics_scores["late_mass"].detach().cpu().tolist())
        mid_scores.extend(physics_scores["midrate_score"].detach().cpu().tolist())
        background_scores.extend(physics_scores["background_mass"].detach().cpu().tolist())
        entropies.extend(entropy_vals.detach().cpu().tolist())
        pearsons.extend(pearson_vals.detach().cpu().tolist())
        ious.extend(iou_vals.detach().cpu().tolist())

        for i in range(len(selected_labels)):
            lbl = selected_labels[i].item()

            if lbl == 2:
                poor_early.append(physics_scores["early_mass"][i].item())
            elif lbl == 0:
                good_late.append(physics_scores["late_mass"][i].item())
            elif lbl == 1:
                midrate_vals.append(physics_scores["midrate_score"][i].item())

    def mean_or_zero(values_list):
        return float(np.mean(values_list)) if values_list else 0.0

    summary_dict = {
        "early_mass_all": mean_or_zero(early_scores),
        "late_mass_all": mean_or_zero(late_scores),
        "midrate_all": mean_or_zero(mid_scores),
        "poor_early_mass": mean_or_zero(poor_early) if poor_early else 0,
        "good_late_mass": mean_or_zero(good_late) if good_late else 0,
        "midrate_score": mean_or_zero(midrate_vals) if midrate_vals else 0,
        "entropy": mean_or_zero(entropies),
        "pearson_corr": mean_or_zero(pearsons),
        "iou_top20": mean_or_zero(ious),
    }

    return sample_rows, summary_dict


def count_parameters(model):
    """Count model trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def training_step(student_model, teacher_model, inputs, labels, optimizer, criterion_cls, lambda_guidance):
    """
    Unified training step: student forward + EMA teacher forward for GGAS

    Returns: (total_loss, l_cls, l_guide)
    """
    inputs = inputs.to(device)
    labels = labels.to(device)

    # === student branch: decide whether to apply attention based on ATTENTION ===
    outputs, feat2_s, attn2_logits_s, _ = student_model(
        inputs,
        enable_attention=(ATTENTION is not None),
        return_guidance=True
    )
    l_cls = criterion_cls(outputs, labels)

    # === teacher branch: use EMA teacher to generate Grad-CAM guidance map G ===
    # Guidance is computed only when LAMBDA_GUIDANCE > 0 and ATTENTION == "ASA"
    l_guide = torch.tensor(0.0, device=device)
    conf_mask = torch.tensor(0.0, device=device)

    if lambda_guidance > 0 and ATTENTION == "ASA": 
        if attn2_logits_s is None:
            raise RuntimeError("attn2_logits_s is None. ATTENTION==\"ASA\" should produce ASA2 attention.")

        gradcam_info = compute_teacher_gradcam_map(teacher_model, inputs)
        guidance_map_G = gradcam_info["guidance_map"]
        conf_mask = gradcam_info["conf_mask"]

        # Size safety check
        assert guidance_map_G.shape == attn2_logits_s.shape, \
            f"Shape mismatch: guidance_map_G {guidance_map_G.shape} vs attn2_logits_s {attn2_logits_s.shape}"

        # ===== Improvement 2: use KL divergence to match spatial distributions instead of MSE on absolute values =====
        # Convert attention logits and guidance map into distributions, then compute KL divergence
        b = attn2_logits_s.shape[0]

        # flatten spatial dimensions
        attn_flat = attn2_logits_s.view(b, -1)
        g_flat = guidance_map_G.view(b, -1)

        # Convert to probability distributions (softmax over spatial dimensions)
        attn_prob = F.softmax(attn_flat, dim=1)
        g_prob = F.softmax(g_flat, dim=1)

        # KL divergence: KL(student || teacher) - student distribution approximates teacher distribution
        # Important: detach teacher target g_prob, keep student attn_prob for gradients
        # Direction: log(attn_prob) is input, g_prob is target
        guide_raw = F.kl_div(
            torch.log(attn_prob + 1e-8),
            g_prob.detach(),
            reduction="none"
        ).sum(dim=1)

        # Use conf_mask to filter low-confidence samples, and use sum/denom to avoid dilution by zeros
        l_guide = (guide_raw * conf_mask).sum() / (conf_mask.sum() + 1e-6)

    total_loss = l_cls + lambda_guidance * l_guide

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Update EMA teacher (unconditional update, so teacher tracks student even during warmup)
    update_ema_teacher(student_model, teacher_model, EMA_DECAY)

    return total_loss, l_cls.detach(), l_guide.detach(), conf_mask.mean().detach()


def train_with_guidance(train_loader, val_loader, fixed_epochs):
    model = LightweightVDLNet_PlacementAblation(
        num_classes=NUM_CLASSES,
        attention=ATTENTION,
    ).to(device)

    # create EMA teacher model
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

        # timing and GPU memory stats
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        # accumulators for GGAS statistics
        sum_cls = 0.0
        sum_guide = 0.0
        sum_conf = 0.0
        num_batches = 0

        # warmup: first WARMUP_EPOCHS epochs disable guidance, classification only
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
            # accumulate GGAS statistics
            sum_cls += l_cls.item()
            sum_guide += l_guide.item()
            sum_conf += conf_mean.item()
            num_batches += 1

            loop.set_postfix(train_loss=f"{running_loss / max(1, num_batches):.4f}")

        # print GGAS statistics at end of epoch
        mean_cls = mean_guide = mean_lambda_guide = mean_conf = mean_train_loss = 0.0
        if num_batches > 0:
            mean_cls = sum_cls / num_batches
            mean_guide = sum_guide / num_batches
            mean_lambda_guide = effective_lambda * mean_guide
            mean_conf = sum_conf / num_batches
            mean_train_loss = running_loss / num_batches


        # calculate training time and memory usage
        epoch_time = time.time() - start_time
        # peak memory in megabytes
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

        val_result = evaluate(model, val_loader, criterion_cls)
        print(
            f"[Epoch {epoch + 1:03d}/{fixed_epochs:03d}] "
            f"guidance={guidance_status} "
            f"lambda={effective_lambda:.4g} | "
            f"train_loss={mean_train_loss:.4f} | "
            f"cls_loss={mean_cls:.4f} | "
            f"guide_loss={mean_guide:.4f} | "
            f"lambda*guide={mean_lambda_guide:.4f} | "
            f"conf={mean_conf:.4f} | "
            f"val_loss={val_result['loss']:.4f} | "
            f"val_acc={val_result['acc']:.2f}% | "
            f"time={epoch_time:.2f}s | "
            f"gpu={peak_mem_mb:.1f}MB"
        )
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "guidance_status": guidance_status,
                "effective_lambda": effective_lambda,
                "train_loss": mean_train_loss,
                "cls_loss": mean_cls,
                "guide_loss": mean_guide,
                "lambda_guide_loss": mean_lambda_guide,
                "conf_mean": mean_conf,
                "val_loss": val_result["loss"],
                "val_acc": val_result["acc"],
                "time_per_epoch_sec": epoch_time,
                "gpu_mem_mb": peak_mem_mb,
            }
        )

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            best_val_state = copy.deepcopy(model.state_dict())

    return {
        "epoch_metrics": epoch_metrics,
        "best_val_state": best_val_state,
    }


def inner_loo_select_best_epoch(outer_train_wells, train_transform, eval_transform, fixed_epochs):
    inner_histories = []

    for inner_val_well in outer_train_wells:
        inner_train_wells = [w for w in outer_train_wells if w != inner_val_well]

        train_ds = CementVDLDataset(DATA_DIR, transform=train_transform, selected_wells=inner_train_wells)
        val_ds = CementVDLDataset(DATA_DIR, transform=eval_transform, selected_wells=[inner_val_well])

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


def train_full_outer_and_test(outer_train_wells, outer_test_well, train_transform, eval_transform, best_epoch):
    # replicate tagging so names inside this function are valid
    att_name = ATTENTION if ATTENTION is not None else "None"
    placement_tag = f"att{att_name}_L1{int(ATT_L1)}_L2{int(ATT_L2)}_L3{int(ATT_L3)}"
    aug_tag = f"flip{int(USE_HFLIP)}_shift{int(USE_TIMESHIFT)}"
    norm_tag = f"norm{NORM_TYPE}"
    full_tag = f"{placement_tag}_{aug_tag}_{norm_tag}_{ASA_PLACEMENT}_{ASA_COMPONENT}_{ASA_FUSION}_hpool{ASA_HPOOL}"

    train_ds = CementVDLDataset(DATA_DIR, transform=train_transform, selected_wells=outer_train_wells)
    test_ds = CementVDLDataset(DATA_DIR, transform=eval_transform, selected_wells=[outer_test_well])

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            f"Outer fold has empty dataset. train={outer_train_wells}, test={[outer_test_well]}"
        )

    train_loader = build_loader(train_ds, is_train=True)
    test_loader = build_loader(test_ds, is_train=False)

    # final training: no validation split, train exactly best_epoch rounds
    model = LightweightVDLNet_PlacementAblation(
        num_classes=NUM_CLASSES,
        attention=ATTENTION,
    ).to(device)

    # create EMA teacher model
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

        # accumulators for GGAS statistics
        sum_cls = 0.0
        sum_guide = 0.0
        sum_conf = 0.0
        num_batches = 0

        # warmup: first WARMUP_EPOCHS epochs disable guidance, classification only
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

            # accumulate GGAS statistics
            sum_cls += l_cls.item()
            sum_guide += l_guide.item()
            sum_conf += conf_mean.item()
            num_batches += 1

        # print GGAS statistics at end of epoch
        if num_batches > 0:
            mean_cls = sum_cls / num_batches
            mean_guide = sum_guide / num_batches
            mean_lambda_guide = effective_lambda * mean_guide
            mean_conf = sum_conf / num_batches

            print(
                f"[Final Epoch {epoch + 1:03d}/{best_epoch:03d}] "
                f"guidance={guidance_status} "
                f"lambda={effective_lambda:.4g} | "
                f"cls_loss={mean_cls:.4f} | "
                f"guide_loss={mean_guide:.4f} | "
                f"lambda*guide={mean_lambda_guide:.4f} | "
                f"conf={mean_conf:.4f}"
            )


    # save model
    att_name = ATTENTION if ATTENTION is not None else "None"
    placement_tag = f"att{att_name}_L1{int(ATT_L1)}_L2{int(ATT_L2)}_L3{int(ATT_L3)}"
    model_path = f"guided_model_outer_{outer_test_well}_{full_tag}_lambda{LAMBDA_GUIDANCE}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model: {model_path}")

    test_result = evaluate(model, test_loader, criterion_cls)
    explain_samples, explain_summary = evaluate_explainability(
        model,
        teacher_model,
        test_loader,
        correct_only=True,
    )
    explainability_df = pd.DataFrame(explain_samples)
    explainability_path = f"explainability_samples_outer_{outer_test_well}_{full_tag}.csv"
    explainability_df.to_csv(explainability_path, index=False)
    print(f"Saved explainability samples: {explainability_path}")
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
        "physics_poor_early": explain_summary["poor_early_mass"],
        "physics_good_late": explain_summary["good_late_mass"],
        "physics_midrate": explain_summary["midrate_score"],
        "attention_entropy": explain_summary["entropy"],
        "attention_corr": explain_summary["pearson_corr"],
        "attention_iou": explain_summary["iou_top20"],
    }


# ==============================================================================
# Main: Two-level LOO
# ==============================================================================


def main():
    print(f"Running attention = {ATTENTION}")
    train_transform = get_transform(is_train=True)
    eval_transform = get_transform(is_train=False)
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

    # print model parameter count
    model = LightweightVDLNet_PlacementAblation(
        num_classes=NUM_CLASSES,
        attention=ATTENTION,
    ).to(device)
    params = count_parameters(model)
    print(f"Model Parameters: {params / 1e3:.2f} K")

    print(
        f"Config: ATTENTION={ATTENTION}, INNER_EPOCHS={INNER_EPOCHS}, "
        f"LAMBDA_GUIDANCE={LAMBDA_GUIDANCE}, "
        f"USE_WEIGHTED_SAMPLER={USE_WEIGHTED_SAMPLER}, "
        f"USE_CLASS_WEIGHT_IN_LOSS={USE_CLASS_WEIGHT_IN_LOSS}, "
        f"USE_FOCAL_LOSS={USE_FOCAL_LOSS}, "
        f"ATT_L1={ATT_L1},ATT_L2={ATT_L2},ATT_L3={ATT_L3}, "
        f"USE_HFLIP={USE_HFLIP}, "
        f"USE_TIMESHIFT={USE_TIMESHIFT}, "
        f"TIMESHIFT_MAX={TIMESHIFT_MAX}, "
        f"NORM_TYPE={NORM_TYPE}, "
        f"ASA_PLACEMENT={ASA_PLACEMENT}, "
        f"ASA_COMPONENT={ASA_COMPONENT}, "
        f"ASA_FUSION={ASA_FUSION}, "
        f"ASA_HPOOL={ASA_HPOOL}"
    )

    if USE_WEIGHTED_SAMPLER and USE_CLASS_WEIGHT_IN_LOSS:
        raise RuntimeError(
            "Both weighted sampler and weighted classification loss are enabled. "
            "Disable one to avoid double class reweighting."
        )

    outer_results = []

    # placement tag used in filenames
    att_name = ATTENTION if ATTENTION is not None else "None"
    placement_tag = f"att{att_name}_L1{int(ATT_L1)}_L2{int(ATT_L2)}_L3{int(ATT_L3)}"
    aug_tag = f"flip{int(USE_HFLIP)}_shift{int(USE_TIMESHIFT)}"
    norm_tag = f"norm{NORM_TYPE}"
    full_tag = f"{placement_tag}_{aug_tag}_{norm_tag}_{ASA_PLACEMENT}_{ASA_COMPONENT}_{ASA_FUSION}_hpool{ASA_HPOOL}"

    for outer_test_well in all_wells:
        print("\n" + "=" * 90)
        print(f"Outer fold start: test well = {outer_test_well}")
        outer_train_wells = [w for w in all_wells if w != outer_test_well]
        print(f"Outer train wells: {outer_train_wells}")

        best_epoch, inner_curve, inner_histories = inner_loo_select_best_epoch(
            outer_train_wells=outer_train_wells,
            train_transform=train_transform,
            eval_transform=eval_transform,
            fixed_epochs=INNER_EPOCHS,
        )
        print(f"Selected best epoch for outer test={outer_test_well}: {best_epoch}")

        att_name = ATTENTION if ATTENTION is not None else "None"
        inner_curve_path = f"inner_curve_outer_test_{outer_test_well}_{full_tag}.csv"
        inner_curve.to_csv(inner_curve_path, index=False)
        print(f"Saved inner-LOO epoch curve: {inner_curve_path}")

        final_result = train_full_outer_and_test(
            outer_train_wells=outer_train_wells,
            outer_test_well=outer_test_well,
            train_transform=train_transform,
            eval_transform=eval_transform,
            best_epoch=best_epoch,
        )

        print(f"Outer test acc ({outer_test_well}): {final_result['test_acc']:.2f}%")
        print("Outer test classification report:")
        print(final_result["report_text"])
        print("Outer test confusion matrix:")
        print(final_result["confusion_matrix"])

        # collect training time and memory usage (use inner_histories rather than inner_curve)
        all_times = []
        all_mems = []

        for hist in inner_histories:
            all_times.extend(hist["time_per_epoch_sec"])
            all_mems.extend(hist["gpu_mem_mb"])

        avg_epoch_time = np.mean(all_times)
        avg_gpu_mem = np.max(all_mems)

        outer_results.append(
            {
                "outer_test_well": outer_test_well,
                "outer_train_wells": ",".join(outer_train_wells),
                "best_epoch": best_epoch,
                "test_acc": final_result["test_acc"],
                "test_loss": final_result["test_loss"],
                "physics_poor_early": final_result["physics_poor_early"],
                "physics_good_late": final_result["physics_good_late"],
                "physics_midrate": final_result["physics_midrate"],
                "attention_entropy": final_result["attention_entropy"],
                "attention_corr": final_result["attention_corr"],
                "attention_iou": final_result["attention_iou"],
                "train_time_epoch_sec": avg_epoch_time,
                "gpu_memory_mb": avg_gpu_mem,
            }
        )
        
        print(f"[PERF] {outer_test_well} | time/epoch={avg_epoch_time:.2f}s | gpu_mem={avg_gpu_mem:.2f}MB")

    results_df = pd.DataFrame(outer_results)
    # placement_tag already computed above
    results_df.to_csv(f"outer_loo_results_{full_tag}.csv", index=False)

    print("\n" + "=" * 90)
    print("Two-level LOO complete.")
    print(results_df)
    print(f"Mean outer test acc: {results_df['test_acc'].mean():.2f}%")
    print(f"Saved summary to outer_loo_results_{full_tag}.csv")


if __name__ == "__main__":
    main()
