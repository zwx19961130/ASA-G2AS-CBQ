import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy  # 导入copy模块用于保存最佳模型
import pandas as pd  # To save classification reports to CSV
import random  # 导入random模块

# ==============================================================================
# 设置随机种子 (Set Random Seeds) - 添加到这里！
# ==============================================================================


def set_seed(seed):
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch Multi-GPU
    np.random.seed(seed)  # NumPy
    random.seed(seed)  # Python built-in random module
    torch.backends.cudnn.deterministic = True  # 确保使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭CUDNN benchmark，它可能引入不确定性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子


SEED = 42  # 定义你的固定种子
set_seed(SEED)

# ==============================================================================
# 配置参数 (Configurations)
# ==============================================================================
DATA_DIR = "vdl_slices_20px"
BATCH_SIZE = 32
NUM_CLASSES = 3
EPOCHS = 100  # 增加epoch数量，因为有早停机制，不必担心过拟合
LEARNING_RATE = 2e-5  # 降低初始学习率
IMAGE_SIZE = (500, 40)
VAL_SPLIT = 0.16
TEST_SPLIT = 0.20

# 更新了模型和结果的保存路径，以反映新的ResNet18架构
MODEL_SAVE_PATH = "resnet18_asa_model_final.pt"
BEST_MODEL_SAVE_PATH = "resnet18_asa_model_best.pt"  # 新增：最佳模型保存路径
CONFUSION_PATH = "confusion_matrix_val_resnet18_asa.png"
CONFUSION_TEST_PATH = "confusion_matrix_test_resnet18_asa.png"
ACC_PLOT_PATH = "accuracy_curve_resnet18_asa.png"

# 早停机制参数
PATIENCE = 30  # 早停的耐心值，连续10个epoch验证损失不下降则停止
MIN_DELTA = 1e-6  # 最小改进，小于此值不认为有提升

# 标签映射
label_map = {"Good": 0, "Midrate": 1, "Poor": 2}
label_names = ["Good", "Midrate", "Poor"]

# ==============================================================================
# 数据集定义 (Dataset Definition)
# ==============================================================================


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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("L")  # 转换为灰度图
        if self.transform:
            image = self.transform(image)
        return image, label


# ==============================================================================
# 数据加载 (Data Loading)
# ==============================================================================
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

full_dataset = CementVDLDataset(DATA_DIR, transform=transform)
total_size = len(full_dataset)
test_size = int(total_size * TEST_SPLIT)
val_size = int(total_size * VAL_SPLIT)
train_size = total_size - val_size - test_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ==============================================================================
# 【核心创新模块】Anisotropic Spatial Attention (ASA)
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


# ==============================================================================
# 【全新模型】LightweightVDLNet_PlacementAblation with Anisotropic Spatial Attention (ASA)
# 用于消融实验，控制ASA模块放置位置
# ==============================================================================
class LightweightVDLNet_PlacementAblation(nn.Module):
    def __init__(self, num_classes=3, apply_asa=True, asa_before_pool=True):
        super(LightweightVDLNet_PlacementAblation, self).__init__()
        self.apply_asa = apply_asa
        self.asa_before_pool = asa_before_pool  # 添加一个标志来控制ASA的放置位置

        # --- Block 1 ---
        self.block1_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block1_pool = nn.MaxPool2d(2)

        # --- Block 2 ---
        self.block2_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block2_pool = nn.MaxPool2d(2)

        # --- Block 3 ---
        self.block3_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block3_pool = nn.MaxPool2d(2)

        # ASA modules
        if self.apply_asa:
            self.asa1 = AnisotropicSpatialAttention(32)
            self.asa2 = AnisotropicSpatialAttention(64)
            self.asa3 = AnisotropicSpatialAttention(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 注意：这里的输入特征数要根据最终特征图的通道数确定
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # 从128维降到64维
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 加入Dropout增强正则化
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # --- Block 1 ---
        x = self.block1_conv(x)
        # if self.apply_asa and self.asa_before_pool:
        #     # 在池化前应用ASA
        #     x = x * self.asa1(x).expand_as(x)
        x = self.block1_pool(x)
        if self.apply_asa and not self.asa_before_pool:
            # 在池化后应用ASA (原始设置)
            x = x * self.asa1(x).expand_as(x)

        # --- Block 2 ---
        x = self.block2_conv(x)
        if self.apply_asa and self.asa_before_pool:
            # 在池化前应用ASA
            x = x * self.asa2(x).expand_as(x)
        x = self.block2_pool(x)
        if self.apply_asa and not self.asa_before_pool:
            # 在池化后应用ASA (原始设置)
            x = x * self.asa2(x).expand_as(x)

        # --- Block 3 ---
        x = self.block3_conv(x)
        # if self.apply_asa and self.asa_before_pool:
        #     # 在池化前应用ASA
        #     x = x * self.asa3(x).expand_as(x)
        x = self.block3_pool(x)
        if self.apply_asa and not self.asa_before_pool:
            # 在池化后应用ASA (原始设置)
            x = x * self.asa3(x).expand_as(x)

        # --- Classifier ---
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def plot_precision_recall_curve(y_true, y_probs, num_classes, label_names, title, save_path):
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        # 获取当前类别的真实标签 (one-hot编码)
        y_true_class = (np.array(y_true) == i).astype(int)
        # 获取当前类别的预测概率
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


# ==============================================================================
# 模型、损失函数和优化器初始化 (Initialization)
# ==============================================================================
# 修改这里来实例化新的模型类并控制ASA的放置位置
# 默认情况下，asa_before_pool=True，即ASA在池化前
model = LightweightVDLNet_PlacementAblation(
    apply_asa=True, asa_before_pool=True)
# 如果你想进行消融实验，测试ASA在池化后，可以这样实例化：
# model = LightweightVDLNet_PlacementAblation(apply_asa=True, asa_before_pool=False)
# 如果你想进行消融实验，测试不使用ASA，可以这样实例化：
# model = LightweightVDLNet_PlacementAblation(apply_asa=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("="*60)
print("Model Architecture: LightweightVDLNet_PlacementAblation with Anisotropic Spatial Attention (ASA)")
print("="*60)
print(model)

# 计算类别权重以处理数据不平衡问题
labels_array = [label for _, label in full_dataset.samples]
class_weights_np = compute_class_weight(
    class_weight='balanced', classes=np.unique(labels_array), y=labels_array)
class_weights = torch.tensor(
    class_weights_np, dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 优化器中增加权重衰减 (Weight Decay)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)  # 增加权重衰减

# 引入学习率调度器 (ReduceLROnPlateau)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-10)

# ==============================================================================
# 训练与验证循环 (Training & Validation Loop)
# ==============================================================================

train_acc_history, val_acc_history = [], []
train_loss_history, val_loss_history = [], []
best_val_loss = float('inf')  # 记录最佳验证损失
patience_counter = 0  # 早停计数器
best_model_wts = None  # 用于保存最佳模型权重
best_val_preds, best_val_labels, best_val_probs = [], [], []  # 存储最佳模型在验证集上的结果

print("\nStarting training with LightweightVDLNet_PlacementAblation...")
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    train_loader_tqdm = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training")
    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_loader_tqdm.set_postfix(
            loss=train_loss/len(train_loader_tqdm), acc=100. * train_correct / train_total)

    # --- 验证 ---
    val_loss, val_correct, val_total = 0, 0, 0
    current_val_preds, current_val_labels = [], []
    current_val_probs = []  # 用于PR曲线的预测分数
    model.eval()
    val_loader_tqdm = tqdm(
        val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Validation")
    with torch.no_grad():
        for inputs, labels in val_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            # 收集概率 (Softmax用于将logits转换为概率)
            probabilities = torch.softmax(outputs, dim=1)
            current_val_probs.extend(probabilities.cpu().numpy())

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            current_val_preds.extend(predicted.cpu().numpy())
            current_val_labels.extend(labels.cpu().numpy())
            val_loader_tqdm.set_postfix(
                loss=val_loss/len(val_loader_tqdm), acc=100. * val_correct / val_total)

    avg_val_loss = val_loss / len(val_loader)
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    train_loss_history.append(train_loss / len(train_loader))
    val_loss_history.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # 学习率调度器步进
    scheduler.step(avg_val_loss)

    # 获取并打印当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {current_lr:.7e}")  # 打印学习率

    # 步骤一：最佳模型保存和早停机制
    if avg_val_loss < best_val_loss - MIN_DELTA:  # 只有当验证损失有显著降低时才更新
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())  # 深拷贝当前模型权重
        patience_counter = 0  # 重置耐心计数器
        # 同时保存最佳模型在验证集上的预测结果
        best_val_preds = current_val_preds
        best_val_labels = current_val_labels
        best_val_probs = current_val_probs
        print(
            f"Validation loss improved. Saving best model to {BEST_MODEL_SAVE_PATH}")
        torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)  # 保存最佳模型
    else:
        patience_counter += 1
        print(
            f"Validation loss did not improve. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break  # 触发早停，终止训练

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
        "classification_report_val_resnet18_asa.csv", index=True)
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
        "classification_report_val_resnet18_asa.csv", index=True)
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
    "pr_curve_val_resnet18_asa.png"
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
    "classification_report_test_resnet18_asa.csv", index=True)
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
    "pr_curve_test_resnet18_asa.png"
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
