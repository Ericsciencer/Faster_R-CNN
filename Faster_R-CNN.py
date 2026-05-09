# MobileNet作为骨干
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from xml.etree import ElementTree as ET
from collections import defaultdict

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# ------------------------------
# 0. 配置
# ------------------------------
DATA_ROOT = "VOC2007_tiny"
NUM_CLASSES = 4                           # 背景 + bird + cat + dog
CLASS_NAMES = ["background", "bird", "cat", "dog"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
BATCH_SIZE = 4
NUM_EPOCHS = 20
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "."
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ------------------------------
# 1. 数据集定义（直接返回Tensor图像）
# ------------------------------
class VOC2007TinyDataset(Dataset):
    def __init__(self, root, image_set="train"):
        self.root = root
        imgset_file = os.path.join(root, "ImageSets", "Main", image_set + ".txt")
        with open(imgset_file, 'r') as f:
            self.img_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.root, "JPEGImages", img_id + ".jpg")
        ann_path = os.path.join(self.root, "Annotations", img_id + ".xml")

        img = F.to_tensor(Image.open(img_path).convert("RGB"))
        boxes, labels = self._parse_annotation(ann_path)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([hash(img_id) % (2**31)])

        return img, target

    def _parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[name]
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
        return boxes, labels

# ------------------------------
# 2. 数据加载器
# ------------------------------
def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloaders():
    dataset = VOC2007TinyDataset(DATA_ROOT, image_set="train")
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    return train_loader, val_loader

# ------------------------------
# 3. 模型：改用 MobileNet V3-Large + FPN
# ------------------------------
def get_model():
    # 使用 torchvision 自带的轻量化 Faster R-CNN
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=NUM_CLASSES)
    return model

# ------------------------------
# 4. 评估函数（mAP）
# ------------------------------
def calculate_ap(recalls, precisions):
    ap = 0.
    for t in np.linspace(0, 1, 11):
        prec = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.
        ap += prec / 11.
    return ap

def evaluate_map(model, data_loader, iou_threshold=0.5):
    model.eval()
    det_results = defaultdict(list)
    gt_boxes_per_class = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating", leave=False):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                img_id = target["image_id"].item()
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()
                for box, label in zip(gt_boxes, gt_labels):
                    gt_boxes_per_class[label.item()][img_id].append(box)

                pred_boxes = output["boxes"].cpu()
                pred_labels = output["labels"].cpu()
                pred_scores = output["scores"].cpu()
                keep = pred_scores > 0.05
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]

                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    det_results[label.item()].append((img_id, score.item(), box))

    aps = []
    for cls in range(1, NUM_CLASSES):
        dets = det_results[cls]
        if not dets:
            aps.append(0.)
            continue

        dets.sort(key=lambda x: x[1], reverse=True)
        gt_dict = gt_boxes_per_class[cls]
        num_gt = sum(len(v) for v in gt_dict.values())
        if num_gt == 0:
            aps.append(0.)
            continue

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        # 为每张图片的 GT 创建对应的匹配状态数组
        gt_matched = {img_id: np.zeros(len(boxes), dtype=bool)
                      for img_id, boxes in gt_dict.items()}

        for i, (img_id, score, pred_box) in enumerate(dets):
            if img_id not in gt_dict:
                fp[i] = 1.
                continue
            gt_boxes = torch.stack(gt_dict[img_id])
            ious = torchvision.ops.box_iou(pred_box.unsqueeze(0), gt_boxes)[0]
            max_iou, max_idx = ious.max(dim=0)
            if max_iou >= iou_threshold and not gt_matched[img_id][max_idx.item()]:
                tp[i] = 1.
                gt_matched[img_id][max_idx.item()] = True
            else:
                fp[i] = 1.

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / num_gt
        precisions = np.divide(tp_cum, (tp_cum + fp_cum),
                               out=np.zeros_like(tp_cum, dtype=float),
                               where=(tp_cum + fp_cum) > 0)
        for i in range(len(precisions)-1, 0, -1):
            precisions[i-1] = max(precisions[i-1], precisions[i])

        ap = calculate_ap(recalls, precisions)
        aps.append(ap)

    return np.mean(aps) if aps else 0.

# ------------------------------
# 5. 训练单 epoch
# ------------------------------
def train_one_epoch(model, optimizer, data_loader, epoch, num_epochs):
    model.train()
    total_loss = 0.
    total_cls_loss = 0.
    total_box_loss = 0.

    pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)
    for images, targets in pbar:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_val = losses.item()
        cls_val = loss_dict["loss_classifier"].item()
        box_val = loss_dict["loss_box_reg"].item()

        total_loss += loss_val
        total_cls_loss += cls_val
        total_box_loss += box_val

        pbar.set_postfix({
            "Loss": f"{loss_val:.4f}",
            "Cls": f"{cls_val:.4f}",
            "Box": f"{box_val:.4f}"
        })

    n = len(data_loader)
    return total_loss / n, total_cls_loss / n, total_box_loss / n

# ------------------------------
# 6. 主流程
# ------------------------------
def main():
    train_loader, val_loader = get_dataloaders()
    model = get_model().to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)

    history = {
        "epoch": [],
        "loss": [],
        "cls_loss": [],
        "box_loss": [],
        "train_map": [],
        "val_map": []
    }

    print("开始训练...")
    for epoch in range(1, NUM_EPOCHS + 1):
        loss_avg, cls_avg, box_avg = train_one_epoch(model, optimizer, train_loader, epoch, NUM_EPOCHS)
        train_map = evaluate_map(model, train_loader)
        val_map = evaluate_map(model, val_loader)

        history["epoch"].append(epoch)
        history["loss"].append(loss_avg)
        history["cls_loss"].append(cls_avg)
        history["box_loss"].append(box_avg)
        history["train_map"].append(train_map)
        history["val_map"].append(val_map)

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Avg Loss: {loss_avg:.4f} "
              f"(cls: {cls_avg:.4f}, box: {box_avg:.4f}) | "
              f"Train mAP: {train_map:.4f} | Val mAP: {val_map:.4f}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "fasterrcnn_mobilenet_voc_tiny.pth"))
    print("模型已保存。")

    # ------------------------------
    # 7. 可视化 loss 与 mAP
    # ------------------------------
    epochs = history["epoch"]

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["loss"], label="Total Loss", marker='o')
    plt.plot(epochs, history["cls_loss"], label="Classification Loss", marker='s')
    plt.plot(epochs, history["box_loss"], label="Box Regression Loss", marker='^')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_map"], label="Train mAP", marker='o')
    plt.plot(epochs, history["val_map"], label="Val mAP", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("mAP Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "map_curve.png"), dpi=150, bbox_inches='tight')
    plt.show()

    # ------------------------------
    # 8. 测试图可视化
    # ------------------------------
    model.eval()
    val_indices = val_loader.dataset.indices
    rand_idx = random.choice(val_indices)
    full_dataset = val_loader.dataset.dataset
    img_tensor, target = full_dataset[rand_idx]

    with torch.no_grad():
        pred = model([img_tensor.to(DEVICE)])[0]

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img_tensor.permute(1, 2, 0).cpu())

    # 真实框（绿色）
    for box, label in zip(target["boxes"], target["labels"]):
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, CLASS_NAMES[label.item()], color='lime', fontsize=10, weight='bold')

    # 预测框（红色，置信度 > 0.5）
    keep = pred["scores"] > 0.5
    boxes = pred["boxes"][keep].cpu()
    labels = pred["labels"][keep].cpu()
    scores = pred["scores"][keep].cpu()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y2+10, f"{CLASS_NAMES[label.item()]}: {score:.2f}",
                color='red', fontsize=10, weight='bold', backgroundcolor='white')

    plt.axis('off')
    plt.title("Test Detection (green: GT, red: prediction)")
    plt.savefig(os.path.join(OUTPUT_DIR, "test_detect.jpg"), dpi=150, bbox_inches='tight')
    plt.show()
    print("可视化完成。")

if __name__ == "__main__":
    main()