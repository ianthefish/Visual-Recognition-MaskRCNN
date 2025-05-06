import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.measure import label
from pycocotools import mask as mask_util
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.rpn import AnchorGenerator
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split
from utils import read_maskfile


def collate_fn(batch):
    return tuple(zip(*batch))


class MedicalSegmentationDataset(Dataset):
    def __init__(self, root_dir, transforms=None, is_test=False, id_map=None):
        self.root = root_dir
        self.transforms = transforms
        self.is_test = is_test
        self.id_map = id_map or {}
        if is_test:
            self.images = sorted(
                [f for f in os.listdir(root_dir) if f.lower().endswith(".tif")]
            )
        else:
            self.images = sorted(
                [
                    d
                    for d in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, d))
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        if self.is_test:
            img = Image.open(os.path.join(self.root, name)).convert("RGB")
            img = (
                self.transforms(img) if self.transforms else T.ToTensor()(img)
            )
            return img, self.id_map.get(name, idx)

        folder = os.path.join(self.root, name)
        img = Image.open(os.path.join(folder, "image.tif")).convert("RGB")
        img = self.transforms(img) if self.transforms else T.ToTensor()(img)

        boxes, masks, labels = [], [], []
        classes = ["class1", "class2", "class3", "class4"]
        for cls_idx, cls_name in enumerate(classes, start=1):
            mp = os.path.join(folder, f"{cls_name}.tif")
            if not os.path.exists(mp):
                continue
            raw = read_maskfile(mp)
            bin_mask = raw > 0
            if not bin_mask.any():
                continue
            lbl = label(bin_mask)

            comps = np.unique(lbl)
            comps = comps[comps != 0]

            for comp in np.unique(lbl):
                if comp == 0:
                    continue
                comp_mask = lbl == comp
                ys, xs = np.where(comp_mask)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                boxes.append([x0, y0, x1, y1])
                masks.append(comp_mask)
                labels.append(cls_idx)

        if not boxes:
            h, w = img.size[1], img.size[0]
            return img, {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, h, w), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,)),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }

        boxes = torch.tensor(boxes, dtype=torch.float32)
        masks = torch.tensor(np.stack(masks), dtype=torch.uint8)
        labels = torch.tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return img, {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }


def get_transform(train=False):
    t = [T.ToTensor()]
    if train:
        t.append(T.RandomHorizontalFlip(0))
    return T.Compose(t)


class BigMaskRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model12 = self._make_submodel(num_classes=3)
        self.model34 = self._make_submodel(num_classes=3)
        self.group12 = [1, 2]
        self.group34 = [3, 4]

    def _make_submodel(self, num_classes):
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(weights=weights)

        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        model.rpn.anchor_generator = anchor_generator
        in_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
        in_fm = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_fm, 256, num_classes
        )
        return model

    def _filter_targets(self, targets, class_ids):
        """
        Keep only labels in class_ids, remap them to 1..len(class_ids).
        e.g. class_ids=[1,2] → labels 1→1, 2→2
             class_ids=[3,4] → labels 3→1, 4→2
        """
        new_t = []
        for t in targets:
            keep_mask = torch.zeros_like(t["labels"], dtype=torch.bool)
            for i, cid in enumerate(class_ids, start=1):
                keep_mask |= t["labels"] == cid
            if not keep_mask.any():
                new_t.append(
                    {
                        "boxes": torch.zeros(
                            (0, 4),
                            dtype=torch.float32,
                            device=t["boxes"].device,
                        ),
                        "labels": torch.zeros(
                            (0,), dtype=torch.int64, device=t["labels"].device
                        ),
                        "masks": torch.zeros(
                            (0, t["masks"].shape[1], t["masks"].shape[2]),
                            dtype=torch.uint8,
                            device=t["masks"].device,
                        ),
                        "image_id": t["image_id"],
                        "area": torch.zeros((0,), device=t["boxes"].device),
                        "iscrowd": torch.zeros(
                            (0,), dtype=torch.int64, device=t["boxes"].device
                        ),
                    }
                )
            else:
                boxes = t["boxes"][keep_mask]
                masks = t["masks"][keep_mask]
                orig_labels = t["labels"][keep_mask]
                new_labels = torch.zeros_like(orig_labels)
                for new_idx, cid in enumerate(class_ids, start=1):
                    new_labels[orig_labels == cid] = new_idx

                new_t.append(
                    {
                        "boxes": boxes,
                        "labels": new_labels,
                        "masks": masks,
                        "image_id": t["image_id"],
                        "area": t["area"][keep_mask],
                        "iscrowd": t["iscrowd"][keep_mask],
                    }
                )
        return new_t

    def forward(self, images, targets=None):
        if self.training:
            t12 = self._filter_targets(targets, self.group12)
            t34 = self._filter_targets(targets, self.group34)

            out12 = self.model12(images, t12)
            out34 = self.model34(images, t34)

            losses = {k: out12[k] + out34[k] for k in out12}
            return losses

        else:
            o12 = self.model12(images)
            o34 = self.model34(images)
            merged = []
            for a, b in zip(o12, o34):
                boxes = torch.cat([a["boxes"], b["boxes"]], dim=0)
                scores = torch.cat([a["scores"], b["scores"]], dim=0)
                masks = torch.cat([a["masks"], b["masks"]], dim=0)
                labels12 = a["labels"]
                labels34 = b["labels"] + 2
                labels = torch.cat([labels12, labels34], dim=0)

                merged.append(
                    {
                        "boxes": boxes,
                        "scores": scores,
                        "labels": labels,
                        "masks": masks,
                    }
                )
            return merged


def generate_submission(
    model, loader, device, output_path="test-results.json", conf_thresh=0.00
):
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, ids in tqdm(loader, desc="Inference"):
            imgs = [i.to(device) for i in imgs]
            outs = model(imgs)
            for img_id, out in zip(ids, outs):
                mid = int(img_id)
                ms = (out["masks"] > 0.5).squeeze(1).cpu().numpy()
                ss = out["scores"].cpu().numpy()
                ls = out["labels"].cpu().numpy()
                bs = out["boxes"].cpu().numpy()
                for m, s, l, b in zip(ms, ss, ls, bs):
                    if s < conf_thresh:
                        continue
                    x1, y1, x2, y2 = b.tolist()
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    rle = mask_util.encode(
                        np.asfortranarray(m.astype(np.uint8))
                    )
                    rle["counts"] = rle["counts"].decode("ascii")
                    results.append(
                        {
                            "image_id": mid,
                            "category_id": int(l),
                            "bbox": bbox,
                            "score": float(s),
                            "segmentation": {
                                "size": rle["size"],
                                "counts": rle["counts"],
                            },
                        }
                    )
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f)
        print(f"Saved predictions to {output_path}")
    return results


def get_eval_subset(full_dataset, num_samples=500):
    indices = np.linspace(0, len(full_dataset) - 1, num_samples, dtype=int)
    return torch.utils.data.Subset(full_dataset, indices)


def main():
    num_epoch = 30
    root = "hw3-data-release"
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("runs/exp")

    train_root = os.path.join(root, "train")

    full_dataset = MedicalSegmentationDataset(
        train_root, get_transform(False), is_test=False
    )

    valid_size = 20
    train_size = len(full_dataset) - valid_size

    train_ds, valid_ds = random_split(full_dataset, [train_size, valid_size])

    print(f"Train size: {len(train_ds)}")
    print(f"Validation size: {len(valid_ds)}")

    train_ds.dataset = MedicalSegmentationDataset(
        train_root, get_transform(True), is_test=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=3,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    eval_loader = DataLoader(
        valid_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = BigMaskRCNN().to(device)

    num_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable parameters: {num_trainable}")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=0)

    best_map = 0.0
    map_metric = MeanAveragePrecision()
    map_metric.warn_on_many_detections = False
    history = {"loss": [], "map": []}

    for epoch in range(1, num_epoch + 1):
        model.train()
        total_loss = 0.0

        print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()[0]:.6f}")

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs = [i.to(device) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(imgs, tgts)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        history["loss"].append(avg_loss)
        writer.add_scalar("Loss/epoch", avg_loss, epoch)

        model.eval()
        map_metric.reset()
        with torch.no_grad():
            for imgs, targets in tqdm(eval_loader, desc=f"Eval Epoch {epoch}"):
                imgs = [i.to(device) for i in imgs]
                preds = model(imgs)
                preds_for_metric, tgts_for_metric = [], []
                for p, t in zip(preds, targets):
                    pred_masks = (p["masks"] > 0.5).squeeze(1).to(torch.uint8)
                    preds_for_metric.append(
                        {
                            "boxes": p["boxes"].cpu(),
                            "scores": p["scores"].cpu(),
                            "labels": p["labels"].cpu(),
                            "masks": pred_masks.cpu(),
                        }
                    )
                    tgts_for_metric.append(
                        {
                            "boxes": t["boxes"],
                            "labels": t["labels"],
                            "masks": t["masks"],
                        }
                    )
                map_metric.update(preds_for_metric, tgts_for_metric)

        res = map_metric.compute()
        history["map"].append(res["map"].item())
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, mAP={res['map']:.4f}")

        if res["map"] > best_map:
            torch.save(model.state_dict(), "checkpoints/best.pth")
            print(f"Saved best model at epoch {epoch}")
            best_map = res["map"].item()

        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pth")

    with open(os.path.join(root, "test_image_name_to_ids.json")) as f:
        infos = json.load(f)
    id_map = {info["file_name"]: info["id"] for info in infos}

    test_ds = MedicalSegmentationDataset(
        os.path.join(root, "test_release"),
        transforms=T.ToTensor(),
        is_test=True,
        id_map=id_map,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    model.load_state_dict(
        torch.load("checkpoints/best.pth", map_location=device)
    )
    generate_submission(model, test_loader, device)

    plt.figure()
    plt.plot(range(1, num_epoch + 1), history["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.savefig("checkpoints/loss.png")
    plt.close()

    plt.figure()
    plt.plot(range(1, num_epoch + 1), history["map"])
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP")
    plt.grid()
    plt.savefig("checkpoints/mAP.png")
    plt.close()


if __name__ == "__main__":
    main()
