from typing import List, Dict, Union
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Evaluate a model and display classification metrics and confusion matrix.

    Args:
        model (nn.Module): Trained model to evaluate.
        dataloader (DataLoader): Evaluation data.
        device (torch.device): CPU or CUDA.
        class_names (List[str]): List of class names.

    Returns:
        Dict[str, Dict[str, Union[float, int]]]: Classification metrics.
    """
    model.eval()
    model.to(device)

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Display classification report as a table
    fig, ax = plt.subplots(figsize=(10, len(report_df) * 0.6))
    ax.axis('off')
    table = ax.table(
        cellText=report_df.round(2).values,
        colLabels=report_df.columns,
        rowLabels=report_df.index,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax.set_title("Classification Report", fontweight="bold")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return report

import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, Dict[str, Union[float, int]]]:
    model.eval()
    model.to(device)

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Per-class accuracy
    class_total = defaultdict(int)
    class_correct = defaultdict(int)
    for true, pred in zip(all_labels, all_preds):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1

    for idx, class_name in enumerate(class_names):
        acc = class_correct[idx] / class_total[idx] if class_total[idx] > 0 else 0.0
        report[class_name]['accuracy'] = acc

    # Global accuracy
    per_class_acc = [metrics[cls]['accuracy'] for cls in class_names]
    support = [metrics[cls]['support'] for cls in class_names]

    macro_avg = sum(per_class_acc) / len(per_class_acc)

    weighted_avg = sum(a * s for a, s in zip(per_class_acc, support)) / sum(support)
    report['macro avg']['accuracy'] = macro_avg
    report['weighted avg']['accuracy'] = weighted_avg

    # Display classification table
    report_df = pd.DataFrame(report).transpose()
    fig, ax = plt.subplots(figsize=(10, len(report_df) * 0.6))
    ax.axis('off')
    table = ax.table(
        cellText=report_df.round(2).values,
        colLabels=report_df.columns,
        rowLabels=report_df.index,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax.set_title("Classification Report", fontweight="bold")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return report

def plot_classification_metrics(metrics: Dict[str, Dict[str, float]], class_names: List[str]) -> None:
    precision = [metrics[cls]["precision"] for cls in class_names]
    recall = [metrics[cls]["recall"] for cls in class_names]
    f1 = [metrics[cls]["f1-score"] for cls in class_names]
    acc = [metrics[cls]["accuracy"] for cls in class_names]
    support = [metrics[cls]["support"] for cls in class_names]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].bar(class_names, support, color='skyblue')
    axes[0].set_title("Support (Samples)")
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(class_names, precision, color='lightgreen')
    axes[1].set_title("Precision")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)

    axes[2].bar(class_names, recall, color='salmon')
    axes[2].set_title("Recall")
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis='x', rotation=45)

    axes[3].bar(class_names, f1, color='mediumpurple')
    axes[3].set_title("F1 Score")
    axes[3].set_ylim(0, 1)
    axes[3].tick_params(axis='x', rotation=45)

    axes[4].bar(class_names, acc, color='orange')
    axes[4].set_title("Per-Class Accuracy")
    axes[4].set_ylim(0, 1)
    axes[4].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = test_dataset.classes

#metrics = evaluate_model(dummy_model, test_loader, device, test_dataset.classes)
# plot_classification_metrics(metrics, test_dataset.classes)