import dataclasses
import pathlib

import torch
import evaluation


@dataclasses.dataclass
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    epochs: int

    train_dataloader: torch.utils.data.DataLoader
    validation_dataloader: torch.utils.data.DataLoader
    test_dataloader: torch.utils.data.DataLoader

    train_evaluator: evaluation.metrics.Metrics
    validation_evaluator: evaluation.metrics.Metrics
    test_evaluator: evaluation.metrics.Metrics

    clearml_logger: evaluation.clearml_logger.ClearMLLogger

    save_path: pathlib.Path


class TrainModel:
    def __init__(self, model_name: str, trainer: Trainer):
        self.model_name = model_name
        self.model = trainer.model
        self.optimizer = trainer.optimizer
        self.criterion = trainer.criterion
        self.epochs = trainer.epochs


        self.train_dataloader = trainer.train_dataloader
        self.validation_dataloader = trainer.validation_dataloader
        self.test_dataloader = trainer.test_dataloader

        self.train_evaluator = trainer.train_evaluator
        self.validation_evaluator = trainer.validation_evaluator
        self.test_evaluator = trainer.test_evaluator

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.best_valuation_metrics = 0.0

        self.clearml_logger = trainer.clearml_logger

        self.save_path = trainer.save_path

    def train(self):
        for epoch in range(self.epochs):

            self.model.train()
            for data, targets in self.train_dataloader:
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.train_evaluator.evaluate(outputs, targets)

            train_accuracy, train_f1, train_precision, train_recall, _ = (
                self.train_evaluator.compute_scores()
            )
            self.train_evaluator.reset_scores()

            self.model.eval()
            with torch.no_grad():
                for data, targets in self.validation_dataloader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.model(data)
                    self.validation_evaluator.evaluate(outputs, targets)

            val_accuracy, val_f1, val_precision, val_recall, _ = (
                self.validation_evaluator.compute_scores()
            )

            self.clearml_logger.log_scalar("Accuracy", "Train", train_accuracy, epoch)
            self.clearml_logger.log_scalar("F1", "Train", train_f1, epoch)
            self.clearml_logger.log_scalar("Precision", "Train", train_precision, epoch)
            self.clearml_logger.log_scalar("Recall", "Train", train_recall, epoch)

            self.clearml_logger.log_scalar(
                "Accuracy", "Validation", val_accuracy, epoch
            )
            self.clearml_logger.log_scalar("F1", "Validation", val_f1, epoch)
            self.clearml_logger.log_scalar(
                "Precision", "Validation", val_precision, epoch
            )
            self.clearml_logger.log_scalar("Recall", "Validation", val_recall, epoch)

            self.validation_evaluator.reset_scores()

            if val_accuracy > self.best_valuation_metrics:
                self.best_valuation_metrics = val_accuracy

                self.save_path.mkdir(parents=True, exist_ok=True)

                model_file = self.save_path / f"{self.model_name}.pt"


                torch.save(self.model.state_dict(), model_file)

                self.clearml_logger.upload_artifact("best_model", model_file)

                print(
                    f"Best model at epoch {epoch + 1} with validation accuracy: {val_accuracy:.4f}"
                )

            print(f"Epoch: {epoch+1}/{self.epochs}\n")

            print(
                f"Train Accuracy: {train_accuracy:.4f}\n"
                f"Train F1: {train_f1:.4f}\n"
                f"Train Precision: {train_precision:.4f}\n"
                f"Train Recall: {train_recall:.4f}\n"
            )

            print(
                f"Validation Accuracy: {val_accuracy:.4f}\n"
                f"Validation F1: {val_f1:.4f}\n"
                f"Validation Precision: {val_precision:.4f}\n"
                f"Validation Recall: {val_recall:.4f}\n"
            )

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for data, targets in self.test_dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                self.test_evaluator.evaluate(outputs, targets)

        test_accuracy, test_f1, test_precision, test_recall, confusion_matrix = (
            self.test_evaluator.compute_scores()
        )
        self.test_evaluator.reset_scores()

        print(
            f"Test Accuracy: {test_accuracy:.4f}\n"
            f"Test F1: {test_f1:.4f}\n"
            f"Test Precision: {test_precision:.4f}\n"
            f"Test Recall: {test_recall:.4f}\n"
            f"Confusion Matrix: {confusion_matrix}"
        )
