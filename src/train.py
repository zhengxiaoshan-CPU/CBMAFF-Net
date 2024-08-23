# train.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from model import create_model


def configure_gpu():
    """
    Configures TensorFlow to use a specific GPU if available.
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("No GPU devices available.")
        return

    try:
        # Select the fourth GPU (index 3) if available
        tf.config.set_visible_devices(physical_devices[3], 'GPU')
        print(f"Using GPU: {physical_devices[3]}")
    except IndexError:
        print("The specified GPU index is out of range.")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")


def load_data():
    """
    Loads and preprocesses the training data.

    Returns:
        tuple: Preprocessed input data and one-hot encoded labels.
    """
    # Load input data
    shift_h_data = np.load('data/train/shiftH.npy')
    shift_h_data = np.expand_dims(shift_h_data, axis=-1)

    type_h_data = np.load('data/train/typeH.npy')
    type_h_data = np.expand_dims(type_h_data, axis=-1)

    type_c_data = np.load('data/train/typeC.npy')
    type_c_data = np.expand_dims(type_c_data, axis=-1)

    shift_c_data = np.load('data/train/shiftC.npy')
    shift_c_data = np.expand_dims(shift_c_data, axis=-1)

    # Load labels
    labels = np.load('data/train/label.npy')

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Convert to one-hot encoding
    one_hot_labels = to_categorical(encoded_labels)

    return shift_c_data, type_c_data, shift_h_data, type_h_data, one_hot_labels


def plot_loss_accuracy(history, save_path='data/pic/loss_accuracy.png'):
    """
    Plots training loss and accuracy over epochs.

    Args:
        history (tensorflow.keras.callbacks.History): History object from model training.
        save_path (str): Path to save the plot image.
    """
    acc = history.history.get('accuracy', [])
    loss = history.history.get('loss', [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 5))
    plt.title('Training Accuracy and Loss')
    plt.plot(epochs, acc, 'r-', label='Training Accuracy')
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', transparent=False)
    plt.show()


def main():
    """
    Main function to execute the training pipeline.
    """
    # Configure GPU settings
    configure_gpu()

    # Load and preprocess data
    shift_c, type_c, shift_h, type_h, labels = load_data()

    # Determine input shapes
    shift_c_shape = shift_c.shape[1:]
    type_c_shape = type_c.shape[1:]
    shift_h_shape = shift_h.shape[1:]
    type_h_shape = type_h.shape[1:]

    # Create the model
    model = create_model(
        shift_c_shape=shift_c_shape,
        type_c_shape=type_c_shape,
        shift_h_shape=shift_h_shape,
        type_h_shape=type_h_shape,
        num_classes=labels.shape[1],
    )

    # Display model summary
    model.summary()

    # Define callbacks
    checkpoint = ModelCheckpoint(
        filepath='data/model/model_best.hdf5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='auto',
    )

    # Train the model
    history = model.fit(
        [shift_c, type_c, shift_h, type_h],
        labels,
        epochs=10,
        batch_size=128,
        callbacks=[checkpoint],
    )

    # Plot training metrics
    plot_loss_accuracy(history, save_path='data/pic/loss_accuracy.png')


if __name__ == '__main__':
    main()
