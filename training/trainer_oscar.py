import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from torch import amp
import torch.backends.cudnn as cudnn
import os
from clearml import Task, OutputModel
from torchmetrics import Accuracy, F1Score, Precision, Recall



cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

task = Task.init(
    project_name="5CCSAGAP Small Group Project (Team 10)",
    task_name="leaf_dr_training(3)",
    output_uri=True
)
task.set_repo(branch='oscar_dev')



logger = task.get_logger()


BATCH_SIZE = 128
LR = 0.001
EPOCHS = 40
IMAGE_SIZE = 256
NUM_CLASSES = 39

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "saved_models", "model1.pth")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

hf_dataset =  load_dataset("DScomp380/plant_village")
hf_split = hf_dataset["train"].train_test_split(test_size=0.2, seed=42)
hf_train = hf_split["train"]

class HFWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform=None):
        self.ds = hf_ds
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"].convert("RGB")
        label = sample["label"]
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = HFWrapper(hf_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

from models.model_oscar import leaf_dr


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")

    model = leaf_dr(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = amp.GradScaler("cuda")
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
    train_f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(device)
    train_precision = Precision(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(device)
    train_recall = Recall(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(device)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            train_accuracy.update(preds, labels)
            train_f1.update(preds, labels)
            train_precision.update(preds, labels)
            train_recall.update(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
                logger.report_scalar("Loss", "Train", loss.item(), batch_idx + epoch * len(train_loader))

        epoch_acc = train_accuracy.compute().item()
        epoch_f1 = train_f1.compute().item()
        epoch_prec = train_precision.compute().item()
        epoch_rec = train_recall.compute().item()

        logger.report_scalar("Accuracy", "Train", epoch_acc, epoch)
        logger.report_scalar("F1", "Train", epoch_f1, epoch)
        logger.report_scalar("Precision", "Train", epoch_prec, epoch)
        logger.report_scalar("Recall", "Train", epoch_rec, epoch)

        train_accuracy.reset()
        train_f1.reset()
        train_precision.reset()
        train_recall.reset()

        scheduler.step()
        logger.report_scalar("Learning Rate", "Train", scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch + 1} completed. LR is now: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    output_model = OutputModel(task=task)
    output_model.update_weights(weights_filename=MODEL_SAVE_PATH, auto_delete_file=False)
    print("Training complete, model saved!")
