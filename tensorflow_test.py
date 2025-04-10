from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), _ = fashion_mnist.load_data()
print("Fashion MNIST train shape:", x_train.shape)
