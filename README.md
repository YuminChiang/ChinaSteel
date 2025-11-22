# Deep Learning Training Pipeline

This project provides a deep learning training pipeline for image classification, targeting the division of images into ten distinct classes. The solution is modular, built around three primary scripts:

- `dataset.py`: Handles dataset loading, preprocessing, and splitting.
- `trainer.py`: Manages model training, validation, and saving checkpoints.
- `main.py`: Entry point for configuring, initializing, and running the full pipeline.

---

## Project Structure

```
ChinaSteel/
├── dataset.py          # Dataset loading and preprocessing
├── trainer.py          # Training, evaluation, checkpoint handling
├── main.py             # Pipeline entry point (CLI, wandb integration)
├── checkpoints/        # Saved model checkpoints
└── README.md           # Documentation
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YuminChiang/ChinaSteel.git
cd ChinaSteel
```

---

## Usage

### 1. Prepare Your Data & Configuration

- Place your dataset in the appropriate folder (specify location in configs/train.yaml or through CLI).
- Adjust training hyperparameters in the `configs/train.yaml` file (or pass them via arguments to `main.py`).

### 2. Run the Training Pipeline

```bash
python main.py
```

**Common Arguments:**
- `--epochs`: Set training epochs
- `--batch_size`: Set batch size
- `--data_path`: Specify dataset location
- `--log_wandb`: Enable wandb logging

### 3. Monitor With wandb

1. Login to your [wandb](https://wandb.ai/) account:
   ```bash
   wandb login
   ```
2. Training metrics, artifacts, and model checkpoints will be automatically logged to wandb for visualization and sharing.

---

## Training Procedure

1. **Dataset Preparation (`dataset.py`):** Loads and processes your dataset, splits into train/val/test sets.
2. **Training (`trainer.py`):** Handles model, optimizer, loss functions, tracking best performance, early stopping, and saving checkpoints.
3. **Pipeline Integration (`main.py`):** Parses configuration, initializes logging, and runs the loop.

---

## Example Outputs

**Console Training Log:**
```
Epoch 1/20 - Train Loss: 0.342 - Val Accuracy: 85.4% | Checkpoint saved
Epoch 2/20 - Train Loss: 0.321 - Val Accuracy: 87.1%
...
```

**Wandb Logging:**
- Live charts for loss and accuracy curves
- Downloadable checkpoints and metrics

**Model Checkpoint:**
- Saved automatically as `checkpoints/model_best.pth` when improved

---

## License

This project is MIT licensed.

## Contact

Questions or contributions? Please open an issue or pull request on GitHub.
