# DL Assignment-1

This repository contains an implementation of a modular **Feedforward Neural Network (FFNN)** built from scratch using only NumPy. The model is trained on both **Fashion MNIST** and **MNIST** datasets and supports experimentation with multiple optimizers, loss functions, weight initialization strategies, and activation functions. Hyperparameter tuning is performed using **Weights & Biases (W&B)** sweeps.

---

## Folder Structure

```
DL ASSIGNMENT-1/
├── Model1.py               # Main model class: forward, backward, training loop
├── Optimizer.py            # SGD, Momentum, NAG, RMSprop, Adam, Nadam
├── Helper_Functions.py     # Gradient updates and optimizer helper functions
├── Utils.py                # Activation/loss functions, weight init, data utils
├── train.py                # Command-line train script
├── sweep_train.py          # Wandb sweep agent script
├── sweep_config.yaml       # YAML config for wandb sweeps
├── compare.py              # Compare MSE vs Cross Entropy loss
├── best_run.py             # Script to load and evaluate saved model
├── requirements.txt        # Python dependencies
├── tests/                  # Unit tests for utility functions
│   ├── test_utils.py
│   ├── test_helper.py
│   └── test_optimizer.py
├── model_final.p           # Serialized trained model (pickle)
├── question_1.py           # Q1 logic (part of assignment)
├── .gitignore              # Git ignore rules
└── README.md               # ← You're here!
```

---

## Setup Instructions

### Step 1: Clone and Set Up Virtual Environment

```bash
git clone https://github.com/Shreyadhondi/da24m019_da6401_assignment1
cd da6401_assignment1

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate     # On Windows

# Install all dependencies
pip install -r requirements.txt
```

---

## Training the Model with Custom Parameters

Train your model using `train.py` with command-line arguments for full control.

### Example Command:

```bash
python train.py \
  --loss cross_entropy \
  --optimizer adam \
  --epochs 30 \
  --batch_size 64 \
  --activation ReLu \
  --num_layers 3 \
  --learning_rate 0.001
```

### Supported Hyperparameters

| Flag             | Values                                                                 |
|------------------|------------------------------------------------------------------------|
| `--loss`         | `cross_entropy`, `mean_squared_error`                                  |
| `--optimizer`    | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`                   |
| `--activation`   | `ReLu`, `tanh`, `sigmoid`                                               |
| `--num_layers`   | 2, 3, 4, 5                                                              |
| `--batch_size`   | 16, 32, 64, 128                                                         |
| `--learning_rate`| e.g., 0.001, 0.0001                                                     |
| `--epochs`       | e.g., 5, 10, 30, 40                                                     |

---

## Hyperparameter Tuning via W&B Sweeps

To run a sweep using your own W&B credentials:

```bash
python sweep_train.py \
  --entity <your-wandb-username> \
  --project da24m019_shreya_da6401_assignment1 \
  --count 10
```

Edit `sweep_config.yaml` to modify parameter ranges or strategy (e.g., `grid`, `random`, `bayes`).

---

## Compare Loss Functions: MSE vs Cross Entropy

Run this script to generate performance plots comparing **mean squared error** and **cross entropy**:

```bash
python compare.py
```

Plots will be logged to W&B automatically and also printed to console.

---

## Running Unit Tests

Run unit tests for all utility and optimizer functions:

```bash
pytest tests/
```

---

## Best Run Example

```text
Run Name: hl_4_bs_32_ac_ReLu_opt_nadam_lr_0.00071
Epochs: 10
Number of Hidden Layers: 4
Hidden Layer Size: 64
Weight Decay: 0.0005
Learning Rate: 0.00071
Optimizer: Nadam
Batch Size: 32
Weight Initialization: Xavier
Activation Function: ReLU
```
---
# Link to the wandb report:
https://api.wandb.ai/links/shreyadhondi-indian-institute-of-technology-madras/g82yl21l

