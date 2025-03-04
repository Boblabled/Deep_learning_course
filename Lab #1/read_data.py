import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def make_metrics_plot(ea):
    plt.figure(figsize=(14, 10))
    draw_plot((2, 2, 1), ea,"MulticlassAccuracy/valid", "Validation Accuracy", "Accuracy")
    draw_plot((2, 2, 2), ea, "MulticlassF1Score/valid", "Validation F1-score", "F1-score")
    draw_plot((2, 2, 3), ea, "MulticlassAccuracy/train", "Training Accuracy", "Accuracy")
    draw_plot((2, 2, 4), ea, "MulticlassF1Score/train", "Training F1-score", "F1-score")

    plt.tight_layout()
    plt.savefig("metrics.png")
    plt.show()

def draw_plot(i, ea, label, title, ylabel):
    data = [event.value for event in ea.Scalars(label)]
    print(label, len(data))
    plt.subplot(*i)
    plt.plot(range(0, len(data)), data, label="MulticlassAccuracy/valid")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()


if __name__ == '__main__':
    PATH = r"logs/EfficientNet/version_1\events.out.tfevents.1741028224.yo.16948.0"

    # Загрузка данных TensorBoard
    ea = event_accumulator.EventAccumulator(PATH)
    ea.Reload()

    make_metrics_plot(ea)

    draw_plot((1, 1, 1), ea, "train_loss", "Training Loss", "Loss")
    plt.savefig("train_loss.png")
    plt.show()

    # # Отдельный график для train_loss
    # train_loss_data = get_scalar_data("train_loss")
    # plt.figure()
    # steps, values = zip(*train_loss_data) if train_loss_data else ([], [])
    # plt.plot(steps, values, label="Train Loss", color="orange")
    # plt.xlabel("Step")
    # plt.ylabel("Loss")
    # plt.title("Training Loss")
    # plt.legend()
    # plt.grid()
    # plt.show()