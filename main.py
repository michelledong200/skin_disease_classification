import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from src.data import build_metadata, download_dataset, encode_labels, split_data
from src.dataset import SkinDataset, get_transforms
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_training_curves,
)
from src.model import build_model, freeze_backbone, unfreeze_all
from src.train import eval_epoch, train_epoch


def build_loaders(train_df, val_df):
    train_loader = DataLoader(
        SkinDataset(train_df, get_transforms(train=True)),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.DEVICE != "cpu",
    )
    val_loader = DataLoader(
        SkinDataset(val_df, get_transforms(train=False)),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.DEVICE != "cpu",
    )
    return train_loader, val_loader


def train(model, train_loader, val_loader, num_classes):
    criterion = nn.CrossEntropyLoss()

    # Phase 1: frozen backbone, only head is trained
    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR_HEAD,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.FREEZE_EPOCHS
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        # Phase 2: unfreeze all at FREEZE_EPOCHS + 1 with a lower lr
        if epoch == config.FREEZE_EPOCHS + 1:
            print(f"\n[Epoch {epoch}] Unfreezing backbone — lr → {config.LR_FULL}")
            unfreeze_all(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.LR_FULL)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.EPOCHS - config.FREEZE_EPOCHS
            )

        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        v_preds, v_labels, v_loss = eval_epoch(model, val_loader, criterion, config.DEVICE)
        v_acc = (v_preds == v_labels).mean()

        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(float(v_acc))

        print(
            f"Epoch {epoch:3d}/{config.EPOCHS} | "
            f"Train loss {t_loss:.4f}  acc {t_acc:.3f} | "
            f"Val   loss {v_loss:.4f}  acc {v_acc:.3f}"
        )

        if v_acc > best_val_acc:
            best_val_acc = float(v_acc)
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"           -> best model saved (val acc {best_val_acc:.3f})")

    return train_losses, val_losses, train_accs, val_accs


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    print(f"Device: {config.DEVICE}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nDownloading dataset...")
    dataset_path = download_dataset()

    df = build_metadata(dataset_path)
    print(f"Images found : {len(df)}")

    df, label_encoder = encode_labels(df)
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    print(f"Classes      : {num_classes}")

    train_df, val_df = split_data(df)
    print(f"Train / Val  : {len(train_df)} / {len(val_df)}")

    train_loader, val_loader = build_loaders(train_df, val_df)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes).to(config.DEVICE)

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nTraining for {config.EPOCHS} epochs "
          f"(frozen backbone for first {config.FREEZE_EPOCHS})...")
    train_losses, val_losses, train_accs, val_accs = train(
        model, train_loader, val_loader, num_classes
    )

    # ── Final evaluation with best checkpoint ────────────────────────────────
    print("\nLoading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    criterion = nn.CrossEntropyLoss()
    preds, labels, _ = eval_epoch(model, val_loader, criterion, config.DEVICE)

    metrics = compute_metrics(preds, labels, class_names)
    print(f"\nVal Accuracy : {metrics['accuracy']:.4f}")
    print(f"Weighted F1  : {metrics['f1_weighted']:.4f}")
    print("\nClassification Report:")
    print(metrics["report"])

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nSaving plots...")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(labels, preds, class_names)
    plot_per_class_f1(labels, preds, class_names)

    print(f"\nDone. Plots → {config.PLOTS_DIR}/")


if __name__ == "__main__":
    main()
