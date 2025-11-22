# trainer.py

import os
import torch
import wandb
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, save_dir, class_names):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.class_names = class_names  
        self.best_acc = 0.0
        
        os.makedirs(self.save_dir, exist_ok=True)

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for images, labels in loop:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_targets, all_preds) * 100
        epoch_f1 = f1_score(all_targets, all_preds, average='macro')  
        
        return epoch_loss, epoch_acc, epoch_f1

    @torch.no_grad()
    def _validate(self, loader=None, phase="Val"):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        target_loader = loader if loader else self.val_loader
        
        loop = tqdm(target_loader, desc=f"[{phase}]", leave=False)
        
        for images, labels in loop:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(target_loader)
        epoch_acc = accuracy_score(all_targets, all_preds) * 100
        epoch_f1 = f1_score(all_targets, all_preds, average='macro')
        
        return epoch_loss, epoch_acc, epoch_f1, all_targets, all_preds

    def fit(self, epochs, scheduler=None):
        print(f"Start training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc, train_f1 = self._train_epoch(epoch)
            
            # Val
            val_loss, val_acc, val_f1, _, _ = self._validate(phase="Val")
            
            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            # WandB Logging (加入 F1 Score)
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/f1": train_f1,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/f1": val_f1,
                "lr": current_lr
            })

            print(f"Epoch {epoch} | "
                  f"Train: Loss={train_loss:.4f} Acc={train_acc:.1f}% F1={train_f1:.3f} | "
                  f"Val: Loss={val_loss:.4f} Acc={val_acc:.1f}% F1={val_f1:.3f}")

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                save_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f" -> Saved best model (Acc: {val_acc:.2f}%)")
        
        print("Training Finished.")

    def test(self, test_loader):
        best_path = os.path.join(self.save_dir, "best_model.pth")
        print(f"Loading best model from {best_path} for testing...")
        self.model.load_state_dict(torch.load(best_path))

        test_loss, test_acc, test_f1, targets, preds = self._validate(loader=test_loader, phase="Test")
        
        print(f"Test Result | Loss: {test_loss:.4f} Acc: {test_acc:.2f}% F1: {test_f1:.3f}")
        
        wandb.log({
            "test/loss": test_loss, 
            "test/acc": test_acc,
            "test/f1": test_f1
        })
        
        print("Generating Confusion Matrix for WandB...")
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=targets,
                preds=preds,
                class_names=self.class_names
            )
        })