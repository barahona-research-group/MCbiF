"""Code for CNN and MLP models."""

import numpy as np, matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

SEED = 42

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(SEED)

def compile_cnn(input_shape, learning_rate=0.0005, filters=32, kernel_size=2):
    """Compile a CNN model with given hyperparameters."""

    # Set the random seed for reproducibility
    initializer = tf.keras.initializers.HeNormal(seed=SEED)

    # Build the CNN model
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(
        layers.Conv2D(
            filters,
            (kernel_size, kernel_size),
            activation="relu",
            kernel_initializer=initializer,
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    # model.add(layers.Dense(128, activation="relu", kernel_initializer=initializer))
    model.add(layers.Dense(1, activation="linear", kernel_initializer=initializer))

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["r2_score"],
    )

    return model

def compile_mlp(input_shape, learning_rate=0.0005, n_nodes=32, n_layers=1, dropout_rate=0):
    """Compile a CNN model with given hyperparameters."""

    # Set the random seed for reproducibility
    initializer = tf.keras.initializers.HeNormal(seed=SEED)

    # Build the CNN model
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())
    for _ in range(n_layers):
        model.add(layers.Dense(n_nodes, activation="relu", kernel_initializer=initializer))
    model.add(layers.Dropout(rate=dropout_rate,seed=SEED))
    model.add(layers.Dense(1, activation="linear", kernel_initializer=initializer))

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["r2_score"],
    )

    return model

def plot_history(history, feature_name, test_r2_score, N, M, ylim_loss=10):

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))

    # R2 plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['r2_score'], label='Train R2 Score')
    plt.plot(history.history['val_r2_score'], label='Validation R2 Score')
    plt.title(f'Model R2 Score')
    plt.ylabel('R2 Score')
    plt.ylim(-0.1, 0.7)
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss')
    plt.ylabel('Loss')
    plt.ylim(0, ylim_loss)
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.suptitle(f"N={N}, M={M}, Features: {feature_name}, Test R2 Score: {round(test_r2_score,4)}", fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()

def evaluate_and_plot(model, X_test, y_test, feature_name, history, N, M, ylim_loss=10):
    # Evaluate the model on the test dataset
    test_loss, test_r2_score = model.evaluate(X_test, y_test)
    print(f'Test R2 Score for {feature_name} features: {test_r2_score:.4f}')
    print(f'Test Loss for {feature_name} features: {test_loss:.4f}')

    # Plot history
    plot_history(history, feature_name, test_r2_score, N, M, ylim_loss)