import os

import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import config


def compute_metrics(preds: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    accuracy = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    report = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )
    return {"accuracy": accuracy, "f1_weighted": f1, "report": report}


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
) -> None:
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, train_losses, label="Train")
    ax_loss.plot(epochs, val_losses, label="Val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, train_accs, label="Train")
    ax_acc.plot(epochs, val_accs, label="Val")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    plt.suptitle("Training Curves")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, class_names: list[str]
) -> None:
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(labels, preds)
    n = len(class_names)

    fig_size = max(10, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    tick_fs = max(6, 10 - n // 5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(class_names, fontsize=tick_fs)

    # Annotate cells only when few classes — unreadable beyond ~20
    if n <= 20:
        thresh = cm.max() / 2
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=max(6, 9 - n // 4))

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_per_class_f1(
    labels: np.ndarray, preds: np.ndarray, class_names: list[str]
) -> None:
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    mean_f1 = np.mean(per_class_f1)

    pairs = sorted(zip(class_names, per_class_f1), key=lambda x: x[1])
    names, scores = zip(*pairs)

    fig, ax = plt.subplots(figsize=(10, max(6, len(class_names) * 0.4)))
    colors = ["steelblue" if s >= mean_f1 else "salmon" for s in scores]
    ax.barh(names, scores, color=colors)
    ax.axvline(mean_f1, color="black", linestyle="--", label=f"Mean F1 = {mean_f1:.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 Score (weighted)")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "per_class_f1.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
