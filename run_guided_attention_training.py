# -*- coding: utf-8 -*-
# run_guided_attention_training.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # 引入F模块
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
import pandas as pd
import random

# ==============================================================================
# 几乎所有配置和原始模型代码都保持不变
# ==============================================================================

# 设置随机种子 (Set Random Seeds)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


SEED = 42
set_seed(SEED)

# 配置参数 (Configurations)
DATA_DIR = "vdl_slices_20px"
BATCH_SIZE = 32  # 为了节省显存，可以适当减小BATCH_SIZE，因为梯度计算会增加消耗
NUM_CLASSES = 3
EPOCHS = 100
LEARNING_RATE = 1e-4
IMAGE_SIZE = (500, 40)
VAL_SPLIT = 0.16
TEST_SPLIT = 0.20
PATIENCE = 30
MIN_DELTA = 1e-6

# ==========================================================
# 【新增】指导学习的超参数 (NEW: Hyperparameter for Guided Learning)
# ==========================================================
# 指导损失的权重 lambda
# lambda_guidance 越大，意味着我们越强迫ASA模块去模仿Grad-CAM的结果
# 设置为 0.0 则完全关闭指导，等同于原始训练方式
LAMBDA_GUIDANCE = 0.01

# 更新模型保存路径以反映新的训练方法
MODEL_SAVE_PATH = f"guided_asa_model_final_lambda{LAMBDA_GUIDANCE}.pt"
BEST_MODEL_SAVE_PATH = f"guided_asa_model_best_lambda{LAMBDA_GUIDANCE}.pt"
# Add for plot path
ACC_PLOT_PATH = f"acc_loss_curves_lambda{LAMBDA_GUIDANCE}.png"
# Add for plot path
CONFUSION_PATH = f"confusion_matrix_val_lambda{LAMBDA_GUIDANCE}.png"
# Add for plot path
CONFUSION_TEST_PATH = f"confusion_matrix_test_lambda{LAMBDA_GUIDANCE}.png"

# 标签映射
label_map = {"Good": 0, "Midrate": 1, "Poor": 2}
label_names = ["Good", "Midrate", "Poor"]


class CementVDLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label_name, label_idx in label_map.items():
            class_dir = os.path.join(root_dir, label_name)
            for fname in os.listdir(class_dir):
                if fname.endswith(".png"):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label_idx))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose(
    [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
full_dataset = CementVDLDataset(DATA_DIR, transform=transform)
total_size = len(full_dataset)
test_size = int(total_size * TEST_SPLIT)
val_size = int(total_size * VAL_SPLIT)
train_size = total_size - val_size - test_size
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)  # 使用多线程加速
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4, pin_memory=True)

# ASA模块和模型定义 (与原文件完全相同，无需修改)


class AnisotropicSpatialAttention(nn.Module):
    def __init__(self, in_planes):
        super(AnisotropicSpatialAttention, self).__init__()
        self.horizontal_conv = nn.Conv1d(
            in_planes, in_planes, kernel_size=21, padding=10, groups=in_planes, bias=False)
        self.vertical_conv = nn.Conv1d(
            in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes, bias=False)
        self.fusion_conv = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False)
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
        super(LightweightVDLNet_PlacementAblation, self).__init__()
        self.apply_asa = apply_asa
        self.asa_before_pool = asa_before_pool
        self.block1_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True)
            )
        self.block1_pool = nn.MaxPool2d(2)
        self.block2_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True)
            )
        self.block2_pool = nn.MaxPool2d(2)
        self.block3_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True)
            )
        self.block3_pool = nn.MaxPool2d(2)
        if self.apply_asa:
            self.asa1 = AnisotropicSpatialAttention(32)
            self.asa2 = AnisotropicSpatialAttention(64)
            self.asa3 = AnisotropicSpatialAttention(128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(
            inplace=True), nn.Dropout(0.5), nn.Linear(64, num_classes))

    def forward(self, x):
        # --- Block 1 ---
        # 第1个卷积块，不使用ASA
        x = self.block1_conv(x)
        x = self.block1_pool(x)

        # --- Block 2 ---
        # 第2个卷积块，在池化前使用ASA
        x_b2 = self.block2_conv(x)
        if self.apply_asa and self.asa_before_pool:
            # 应用 asa2
            x = x_b2 * self.asa2(x_b2).expand_as(x_b2)
        else:
            x = x_b2
        x = self.block2_pool(x)

        # --- Block 3 ---
        # 第3个卷积块，不使用ASA
        x = self.block3_conv(x)
        x = self.block3_pool(x)

        # --- Classifier ---
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 模型、损失函数和优化器初始化
model = LightweightVDLNet_PlacementAblation(
    apply_asa=True, asa_before_pool=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
labels_array = [label for _, label in full_dataset.samples]
class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(
    labels_array), y=labels_array), dtype=torch.float32, device=device)
criterion_cls = nn.CrossEntropyLoss(weight=class_weights)  # 分类损失
criterion_guide = nn.MSELoss()  # 指导损失
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-10)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=LEARNING_RATE, # 这里设置你选定的最大学习率
                                                steps_per_epoch=len(train_loader), 
                                                epochs=EPOCHS,
                                                pct_start=0.3) # 30%的时间用于预热


# ==============================================================================
# 【核心创新】训练循环改造：集成注意力监督
# ==============================================================================

# 创建字典来存储hooks捕获的中间层输出
intermediate_results = {
    'pre_asa_features': None,  # ASA模块前的特征图 A
    'asa_attention_map': None  # ASA模块生成的注意力图 S
}

# 定义Hook函数


def get_pre_asa_features_hook(module, input, output):
    """捕获目标模块的输出，即ASA模块的输入特征图"""
    intermediate_results['pre_asa_features'] = output


def get_asa_attention_map_hook(module, input, output):
    """捕获ASA模块自身生成的注意力图"""
    intermediate_results['asa_attention_map'] = output


# 我们选择指导L2的ASA模块
target_conv_block = model.block2_conv  # 特征图A的来源
target_asa_module = model.asa2  # 注意力图S的来源

train_acc_history, val_acc_history = [], []
train_loss_history, val_loss_history = [], []
best_val_loss = float('inf')
patience_counter = 0
best_model_wts = None

# 用于保存最佳验证结果的变量
best_val_labels = []
best_val_preds = []
best_val_probs = []

print(
    f"\nStarting training with ATTENTION GUIDANCE (lambda={LAMBDA_GUIDANCE})...")
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    total_cls_loss = 0
    total_guide_loss = 0

    train_loader_tqdm = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training")

    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)

        # ============ 注册 Hooks ============
        # 在每次迭代开始时，注册hooks来捕获我们需要的中间结果
        handle_feat = target_conv_block.register_forward_hook(
            get_pre_asa_features_hook)
        handle_attn = target_asa_module.register_forward_hook(
            get_asa_attention_map_hook)

        # ======== 1. 正常的前向传播 ========
        # 这一次传播会触发hooks，并将A和S存入intermediate_results字典
        outputs = model(inputs)

        # ======== 2. 计算分类损失 ========
        l_cls = criterion_cls(outputs, labels)

        # ======== 3. 计算指导损失 (如果lambda > 0) ========
        if LAMBDA_GUIDANCE > 0:
            # 从字典中获取A和S
            features_A = intermediate_results['pre_asa_features']  # 特征图 A
            attention_S = intermediate_results['asa_attention_map']   # 注意力图 S

            # --- 生成引导图 G ---
            # a. 获取目标类别的预测分数
            # 使用真实标签来指导，也可以使用预测最高分的标签来指导
            target_scores = outputs.gather(1, labels.unsqueeze(1)).squeeze()

            # b. 计算梯度：∂(target_scores)/∂(features_A)
            # 必须设置retain_graph=True，因为我们稍后还要为总损失进行backward
            gradients = torch.autograd.grad(outputs=target_scores.sum(),  # 对整个batch的得分求和再计算梯度
                                            inputs=features_A,
                                            retain_graph=True)[0]

            # c. Grad-CAM核心计算
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            guidance_map_G = torch.sum(
                weights * features_A, dim=1, keepdim=True)  # [B, 1, H, W]
            guidance_map_G = F.relu(guidance_map_G)  # ReLU

            # d. 归一化引导图 G 到 [0, 1] 区间，使其与 S 可比
            # 逐个图像进行归一化，防止batch内极端值影响
            B, _, H, W = guidance_map_G.shape
            g_flat = guidance_map_G.view(B, -1)
            g_min = g_flat.min(dim=1, keepdim=True)[
                0].unsqueeze(-1).unsqueeze(-1)
            g_max = g_flat.max(dim=1, keepdim=True)[
                0].unsqueeze(-1).unsqueeze(-1)
            guidance_map_G = (guidance_map_G - g_min) / \
                (g_max - g_min + 1e-8)  # 加上eps防止除零

            # # --- 【新增代码：提纯指导图】 ---
            # # 1. 确定阈值，例如，保留前20%的激活值
            # flat_map = guidance_map_G.view(B, -1)
            # # 计算每个样本的阈值
            # threshold = torch.quantile(flat_map, 0.8, dim=1, keepdim=True)[
            #     0].unsqueeze(-1).unsqueeze(-1)

            # # 2. 将低于阈值的设为0，高于的设为1，生成一个清晰的目标
            # guidance_map_G_purified = (guidance_map_G >= threshold).float()

            # 3. 用提纯后的图计算损失
            # detach() G, 因为G是“标准答案”，不应通过它来计算梯度
            # l_guide = criterion_guide(attention_S, guidance_map_G.detach())

            # e. 计算 S 和 G 之间的损失
            # detach() G, 因为G是“标准答案”，不应通过它来计算梯度
            l_guide = criterion_guide(attention_S, guidance_map_G.detach())
        else:
            l_guide = torch.tensor(0.0).to(device)  # 如果不使用指导，则损失为0

        # ======== 4. 计算总损失 ========
        total_loss = l_cls + LAMBDA_GUIDANCE * l_guide

        # ======== 5. 反向传播和优化 ========
        
        total_loss.backward()
        optimizer.step()

        scheduler.step() # <--- 在每个批次后更新
        optimizer.zero_grad()  # 推荐的顺序

        # ============ 移除 Hooks ============
        # 在迭代结束后必须移除hooks，防止内存泄漏和不必要的计算
        handle_feat.remove()
        handle_attn.remove()

        # ======== 6. 记录统计数据 ========
        train_loss += total_loss.item()
        total_cls_loss += l_cls.item()
        total_guide_loss += l_guide.item() if isinstance(l_guide, torch.Tensor) else l_guide

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # 更新tqdm进度条
        train_loader_tqdm.set_postfix(
            loss=f"{train_loss/len(train_loader_tqdm):.4f}",
            cls=f"{total_cls_loss/len(train_loader_tqdm):.4f}",
            guide=f"{total_guide_loss/len(train_loader_tqdm):.4f}",
            acc=f"{100.*train_correct/train_total:.2f}%"
        )

    # --- 验证循环 (保持不变) ---
    val_loss, val_correct, val_total = 0, 0, 0
    current_val_labels_epoch, current_val_preds_epoch, current_val_probs_epoch = [
    ], [], []  # for current epoch validation results

    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion_cls(outputs, labels)  # 验证时只关心分类损失
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            # Collect labels, predictions, and probabilities for current validation epoch
            current_val_labels_epoch.extend(labels.cpu().numpy())
            current_val_preds_epoch.extend(predicted.cpu().numpy())
            probabilities = torch.softmax(outputs, dim=1)
            current_val_probs_epoch.extend(probabilities.cpu().numpy())

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total

    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(f"\nEpoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {current_lr:.7e}")
    # scheduler.step(avg_val_loss)

    # 早停和模型保存机制 (保持不变)
    if avg_val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0
        print(
            f"Validation loss improved. Saving best model to {BEST_MODEL_SAVE_PATH}")
        torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)

        # 当验证损失改善时，更新最佳验证结果
        best_val_labels = current_val_labels_epoch
        best_val_preds = current_val_preds_epoch
        best_val_probs = current_val_probs_epoch

    else:
        patience_counter += 1
        print(
            f"Validation loss did not improve. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

# 确保在训练循环结束后，即使没有早停，也有最后一个epoch的验证数据
current_val_labels = current_val_labels_epoch
current_val_preds = current_val_preds_epoch
current_val_probs = current_val_probs_epoch


# 绘制 Precision-Recall 曲线的辅助函数 (从原始文件提取，如果不存在则添加)
def plot_precision_recall_curve(y_true, y_probs, num_classes, class_names, title, save_path):
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(
            np.array(y_true) == i, np.array(y_probs)[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision,
                 label=f'Class {class_names[i]} (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


# 验证集分类报告（使用最佳模型在验证集上的表现）
if best_model_wts:
    print(f"\nGenerating Validation Set Classification Report based on Best Model...")
    report_val_str = classification_report(
        best_val_labels, best_val_preds, target_names=label_names, zero_division=0, digits=5)
    print(
        f"\nValidation Set Classification Report (Best Model):\n{report_val_str}")
    report_val_dict = classification_report(
        best_val_labels, best_val_preds, target_names=label_names, output_dict=True, zero_division=0, digits=5)
    pd.DataFrame(report_val_dict).transpose().to_csv(
        "classification_report_val_resnet18_asa.csv", index=True)  # 可以考虑将lambda值加入文件名
    print(f"Validation set classification report saved to classification_report_val_resnet18_asa.csv")
else:
    # 如果没有早停（例如EPOCHS设置得很小），就用最后一个epoch的验证结果
    print(f"\nGenerating Validation Set Classification Report based on Final Model (No Early Stop)...")
    report_val_str = classification_report(
        current_val_labels, current_val_preds, target_names=label_names, zero_division=0, digits=5)
    print(
        f"\nValidation Set Classification Report (Final Model):\n{report_val_str}")
    report_val_dict = classification_report(
        current_val_labels, current_val_preds, target_names=label_names, output_dict=True, zero_division=0, digits=5)
    pd.DataFrame(report_val_dict).transpose().to_csv(
        "classification_report_val_resnet18_asa.csv", index=True)  # 可以考虑将lambda值加入文件名
    print(f"Validation set classification report saved to classification_report_val_resnet18_asa.csv")

# ==============================================================================
# 保存与评估 (Save & Evaluate)
# ==============================================================================

# 如果有最佳模型，加载最佳模型进行最终评估
if best_model_wts:
    model.load_state_dict(best_model_wts)
    print(
        f"\nLoaded best model from {BEST_MODEL_SAVE_PATH} for final evaluation.")
else:
    # 如果没有早停，也保存最后一次训练的模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nNo early stopping, saving final model to {MODEL_SAVE_PATH}")


plt.figure(figsize=(12, 10))  # Adjust figure size for two subplots
plt.subplot(2, 1, 1)  # First subplot for accuracy

plt.plot(train_acc_history, label="training accuracy",
         marker='o', linestyle='-')
plt.plot(val_acc_history, label="validation accuracy",
         marker='x', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training & Validation Accuracy (LightweightVDLNet_PlacementAblation)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)  # Second subplot for loss

plt.plot(train_loss_history, label="training loss", marker='o', linestyle='-')
plt.plot(val_loss_history, label="validation loss", marker='x', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss (LightweightVDLNet_PlacementAblation)")
plt.legend()
plt.grid(True)

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.savefig(ACC_PLOT_PATH)
plt.close()
print(f"Accuracy and Loss curves saved to {ACC_PLOT_PATH}")

# 绘制验证集混淆矩阵（基于最佳模型所在的epoch的数据）
cm = confusion_matrix(best_val_labels if best_model_wts else current_val_labels,
                      best_val_preds if best_model_wts else current_val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix on Validation Set (LightweightVDLNet_PlacementAblation)")
plt.savefig(CONFUSION_PATH)
plt.close()


# 绘制验证集Precision-Recall曲线
plot_precision_recall_curve(
    best_val_labels if best_model_wts else current_val_labels,
    best_val_probs if best_model_wts else current_val_probs,
    NUM_CLASSES, label_names,
    "Precision-Recall Curve on Validation Set (LightweightVDLNet_PlacementAblation)",
    # Changed filename to include lambda
    f"pr_curve_val_lambda{LAMBDA_GUIDANCE}.png"
)


# 测试集评估
model.eval()
test_preds, test_labels = [], []
all_test_probs = []  # 用于PR曲线的预测分数
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        # 收集概率
        probabilities = torch.softmax(outputs, dim=1)
        all_test_probs.extend(probabilities.cpu().numpy())

        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())


# 测试集分类报告
report_test_str = classification_report(
    test_labels, test_preds, target_names=label_names, zero_division=0, digits=5)
print(f"\nTest Set Classification Report:\n{report_test_str}")
report_test_dict = classification_report(
    test_labels, test_preds, target_names=label_names, output_dict=True, zero_division=0, digits=5)
pd.DataFrame(report_test_dict).transpose().to_csv(
    "classification_report_test_resnet18_asa.csv", index=True)  # 可以考虑将lambda值加入文件名
print(f"Test set classification report saved to classification_report_test_resnet18_asa.csv")

cm_test = confusion_matrix(test_labels, test_preds)
disp_test = ConfusionMatrixDisplay(
    confusion_matrix=cm_test, display_labels=label_names)
disp_test.plot(cmap=plt.cm.Oranges)
plt.title("Confusion Matrix on Test Set (LightweightVDLNet_PlacementAblation)")
plt.savefig(CONFUSION_TEST_PATH)
plt.close()


# 绘制测试集Precision-Recall曲线
plot_precision_recall_curve(
    test_labels, all_test_probs, NUM_CLASSES, label_names,
    "Precision-Recall Curve on Test Set (LightweightVDLNet_PlacementAblation)",
    # Changed filename to include lambda
    f"pr_curve_test_lambda{LAMBDA_GUIDANCE}.png"
)


test_correct = sum(p == t for p, t in zip(test_preds, test_labels))
test_total = len(test_labels)
test_acc = 100 * test_correct / test_total
print(
    f"Final Test Accuracy (LightweightVDLNet_PlacementAblation): {test_acc:.2f}%")
print("Training and evaluation complete.")


# ==============================================================================
# 导出模型为 ONNX 格式 (Export Model to ONNX)
# ==============================================================================
print("\nAttempting to export model to ONNX format...")
try:
    # 加载最佳模型（如果有的话）
    if best_model_wts:
        model.load_state_dict(best_model_wts)
        print("Loaded best model weights for ONNX export.")
    else:
        print("Using final model weights for ONNX export.")

    model.eval()  # 确保模型处于评估模式

    # 创建一个与模型输入尺寸匹配的虚拟输入
    # IMAGE_SIZE (500, 40) 是 (H, W)，模型期望 (N, C, H, W)
    # N=1 (batch size 1), C=1 (灰度图)
    dummy_input = torch.randn(1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)

    onnx_model_path = "lightweight_vdl_net_asa.onnx"  # 定义ONNX文件路径
    torch.onnx.export(model,                    # 运行的模型
                      dummy_input,               # 模型的一个示例输入 (会根据这个输入追踪计算图)
                      onnx_model_path,           # 模型保存路径
                      export_params=True,        # 导出训练好的参数
                      opset_version=11,          # ONNX操作集版本
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=['input'],     # 输入名称
                      output_names=['output'],   # 输出名称
                      dynamic_axes={'input': {0: 'batch_size'},    # 声明动态批处理大小
                                    'output': {0: 'batch_size'}})
    print(f"Model successfully exported to ONNX format at: {onnx_model_path}")
    print("You can now open this .onnx file with Netron for visualization.")

except Exception as e:
    print(f"Error exporting model to ONNX: {e}")
    print("Please ensure your dummy_input matches the model's expected input shape.")

print("All tasks complete.")


print("\nGuided Attention Training and evaluation complete.")
