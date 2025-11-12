from pathlib import Path
from clearml import Task

from data import augmentation, dataset_manager, hf_dataset
from evaluation import clearml_logger, metrics
from training import trainer
from models import plant_net_model

import torch

def main():

    dataset = dataset_manager.DatasetLoader("DScomp380/plant_village", 1)

    geometric_augmentation = augmentation.GeometricAugmentation()
    resolution_augmentation = augmentation.ResolutionAugmentation()
    photometric_augmentation = augmentation.PhotometricAugmentation()

    dataset_augmentation = augmentation.DatasetAugmentation(geometric_augmentation, resolution_augmentation, photometric_augmentation)

    train_dataset = hf_dataset.HFDataset(dataset.training_split, transform=dataset_augmentation.transform_train_dataset())
    val_dataset = hf_dataset.HFDataset(dataset.validation_split, transform=dataset_augmentation.transform_val_test_dataset())
    test_dataset = hf_dataset.HFDataset(dataset.test_split, transform=dataset_augmentation.transform_val_test_dataset())

    num_classes = 39
    model = plant_net_model.PlantNet(num_classes)
    epochs = 100
    learning_rate = 5e-4
    batch_size = 256
    num_workers = 12
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    save_path = Path("./model_weights")

    hyperparameters = {
        "model": model,
        "num_classes": num_classes,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "optimizer": optimizer,
        "criterion": criterion,

    }

    clearml_log = clearml_logger.ClearMLLogger(
        project_name="5CCSAGAP Small Group Project (Team 10)",
        task_name="Plant Net v1.1",
        task_type=Task.TaskTypes.training,
        params=hyperparameters
    )

    trainer_values = trainer.Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=5e-4, nesterov=True, momentum=0.9),
        criterion=torch.nn.CrossEntropyLoss(),
        epochs=epochs,

        train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=12,shuffle=True),
        validation_dataloader=torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=12, shuffle=False),
        test_dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=12, shuffle=False),

        train_evaluator=metrics.Metrics("multiclass", 39),
        validation_evaluator=metrics.Metrics("multiclass", 39),
        test_evaluator=metrics.Metrics("multiclass", 39),

        clearml_logger=clearml_log,

        save_path=save_path
    )

    train = trainer.TrainModel("plant_net_v1.1", trainer_values)

    train.train()
    train.test()

if __name__ == "__main__":
    main()