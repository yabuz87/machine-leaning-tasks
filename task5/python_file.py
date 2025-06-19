import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
# Load MNIST data
(x_train, y_train), _ = mnist.load_data()

# Normalize and flatten images
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
y_train = to_categorical(y_train, 10)

# Model sizes
input_size = 784
hidden_size = 128
output_size = 10
batch_size = 64
epochs = 7

# Softmax with numerical stability
def softmax(x):
    e_x = tf.exp(x - tf.reduce_max(x, axis=0, keepdims=True))
    return e_x / (tf.reduce_sum(e_x, axis=0, keepdims=True) + 1e-9)

# Initialize weights (TensorFlow variables)
tf.random.set_seed(42)
W1 = tf.Variable(tf.random.normal((hidden_size, input_size), stddev=0.01))
W2 = tf.Variable(tf.random.normal((output_size, hidden_size), stddev=0.01))

# Learning rates
lr_s = 0.1
lr_w = 0.005

# Training loop
for epoch in range(epochs):
    total_correct = 0
    total_samples = 0

    for i in range(0, x_train.shape[0], batch_size):
        # Mini-batch
        x_batch = x_train[i:i+batch_size].T  # shape: [784, batch]
        y_batch = y_train[i:i+batch_size].T  # shape: [10, batch]
        bsz = x_batch.shape[1]

        # Layer states
        x0 = tf.convert_to_tensor(x_batch)
        x1 = tf.zeros((hidden_size, bsz))
        x2 = tf.zeros((output_size, bsz))

        # Predictive coding loop
        for step in range(5):
            pred_x0 = tf.linalg.matmul(tf.transpose(W1), x1)
            pred_x1 = tf.linalg.matmul(tf.transpose(W2), x2)

            err_x0 = x0 - pred_x0
            err_x1 = x1 - pred_x1
            err_x2 = y_batch - softmax(x2)

            # transpose W2 here to avoid shape mismatch
            x1 += lr_s * (tf.linalg.matmul(W1, err_x0) - tf.linalg.matmul(tf.transpose(W2), err_x2))
            x2 += lr_s * (tf.linalg.matmul(W2, err_x1) + err_x2)
               # Weight updates
            W1.assign_add(lr_w * tf.matmul(x1, tf.transpose(err_x0)))  # shape: (128, 784)
            W2.assign_add(lr_w * tf.matmul(x2, tf.transpose(err_x1)))  # shape: (10, 128)



        # Accuracy computation
        preds = tf.argmax(softmax(x2), axis=0).numpy()
        labels = tf.argmax(y_batch, axis=0).numpy()
        acc = np.mean(preds == labels)

        total_correct += acc * bsz
        total_samples += bsz

    print(f"Epoch {epoch+1}: Accuracy = {(total_correct / total_samples) * 100:.2f}%  total_correct={total_correct} and total_samples={total_samples}")
