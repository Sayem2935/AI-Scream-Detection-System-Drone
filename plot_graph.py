import pickle

import matplotlib.pyplot as plt


with open("history.pkl", "rb") as file_obj:
    history = pickle.load(file_obj)


def get_metric(name):
    values = history.get(name)
    val_values = history.get(f"val_{name}")
    return values, val_values


loss, val_loss = get_metric("loss")
accuracy, val_accuracy = get_metric("accuracy")
precision, val_precision = get_metric("precision")
recall, val_recall = get_metric("recall")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()


def plot_metric(ax, train_values, val_values, title, ylabel):
    if train_values is None or val_values is None:
        ax.set_title(f"{title} (not available)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        return

    ax.plot(train_values, label=f"Train {title}")
    ax.plot(val_values, label=f"Validation {title}")
    ax.set_title(f"{title} Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()


plot_metric(axes[0], loss, val_loss, "Loss", "Loss")
plot_metric(axes[1], accuracy, val_accuracy, "Accuracy", "Accuracy")
plot_metric(axes[2], precision, val_precision, "Precision", "Precision")
plot_metric(axes[3], recall, val_recall, "Recall", "Recall")

plt.tight_layout()
plt.show()
