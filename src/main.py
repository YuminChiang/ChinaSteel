# main.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import wandb

from dataset import get_dataloaders
from trainer import Trainer

def main(args):
    # 1. Setup
    wandb.init(project=args.project_name, config=args, name=args.run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU device name:", torch.cuda.get_device_name(0))
    
    # 2. Data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    # 3. Model
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, len(class_names))
    model = model.to(device)

    # 4. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 5. Init Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir,
        class_names=class_names  
    )

    # 6. Run
    trainer.fit(epochs=args.epochs, scheduler=scheduler)
    trainer.test(test_loader)
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../Simple_chinastell_data")
    parser.add_argument("--save_dir", type=str, default="../checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--project_name", type=str, default="DL_hw1)")
    parser.add_argument("--run_name", type=str, default="vgg16")
    
    args = parser.parse_args()
    main(args)