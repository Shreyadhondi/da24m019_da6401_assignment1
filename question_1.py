import numpy as np
import wandb
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Initialize W&B run
wandb.init(project="da24m019_shreya_da6401_assignment1", name="fashion-class-samples")

# Load dataset
(x_train, y_train), (_, _) = fashion_mnist.load_data()

# Pick one image per class
samples = []
for class_id in range(10):
    idx = np.where(y_train == class_id)[0][0]  # Get index of the first occurrence
    img = x_train[idx]
    caption = class_names[class_id]
    samples.append(wandb.Image(img, caption=caption))

# Log to W&B
wandb.log({"examples": samples})
