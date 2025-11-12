import torch
import torchmetrics


class Metrics:
    def __init__(self, task, num_classes):
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = _device
        self.accuracy = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(
            _device
        )
        self.f1_score = torchmetrics.F1Score(
            task=task, num_classes=num_classes, average="macro"
        ).to(_device)
        self.precision = torchmetrics.Precision(
            task=task, num_classes=num_classes, average="macro"
        ).to(_device)
        self.recall = torchmetrics.Recall(
            task=task, num_classes=num_classes, average="macro"
        ).to(_device)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            task=task, num_classes=num_classes
        ).to(_device)

    def evaluate(self, predictions, targets):
        _predictions = torch.argmax(predictions, dim=1)
        _predictions, _targets = _predictions.to(self.device), targets.to(self.device)

        self.accuracy.update(_predictions, _targets)
        self.f1_score.update(_predictions, _targets)
        self.precision.update(_predictions, _targets)
        self.recall.update(_predictions, _targets)
        self.confusion_matrix.update(_predictions, _targets)

    def compute_scores(self):
        return (
            self.accuracy.compute().item(),
            self.f1_score.compute().item(),
            self.precision.compute().item(),
            self.recall.compute().item(),
            self.confusion_matrix.compute(),
        )

    def reset_scores(self):
        self.accuracy.reset()
        self.f1_score.reset()
        self.precision.reset()
        self.recall.reset()
        self.confusion_matrix.reset()
