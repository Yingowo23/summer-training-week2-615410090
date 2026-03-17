import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import wandb

from rfmid_dataset import RFMiDDataset
from focal_loss import FocalLoss


def get_model(model_name: str, num_classes: int):
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "vgg16"])
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="student_diana_week2")
    parser.add_argument("--entity", type=str, default="dianalee5328-national-chung-cheng-university")
    parser.add_argument("--project", type=str, default="summer-training-week2")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 儲存設定
    config_path = os.path.join(args.output_dir, f"{args.model}_{args.loss}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    wandb.init(
        entity=args.entity,
        project=args.project,
        config=vars(args),
        name=f"{args.model}_{args.loss}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = RFMiDDataset(
        csv_file="Retinal-disease-classification/RFMiD_Training_Labels.csv",
        img_dir="Retinal-disease-classification/Training",
        transform=transform
    )

    val_dataset = RFMiDDataset(
        csv_file="Retinal-disease-classification/RFMiD_Validation_Labels.csv",
        img_dir="Retinal-disease-classification/Validation",
        transform=transform
    )

    test_dataset = RFMiDDataset(
        csv_file="Retinal-disease-classification/RFMiD_Testing_Labels.csv",
        img_dir="Retinal-disease-classification/Test",
        transform=transform
    )

    num_classes = len(train_dataset.label_cols)
    print("Detected num_classes:", num_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = get_model(args.model, num_classes).to(device)

    if args.loss == "focal":
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, f"{args.model}_{args.loss}_best.pt")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    wandb.log({
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc
    })

    wandb.finish()


if __name__ == "__main__":
    main()