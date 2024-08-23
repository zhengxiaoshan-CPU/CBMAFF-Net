import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    f1_score, 
    recall_score, 
    precision_score
)
import matplotlib.pyplot as plt
import itertools


def configure_gpu(gpu_index=1):
    """
    Configures TensorFlow to use a specific GPU if available.

    Args:
        gpu_index (int): Index of the GPU to use.
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("No GPU devices available.")
        return

    try:
        tf.config.set_visible_devices(physical_devices[gpu_index], 'GPU')
        print(f"Using GPU: {physical_devices[gpu_index]}")
    except IndexError:
        print("The specified GPU index is out of range.")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")


def load_test_data():
    """
    Loads and preprocesses the test data.

    Returns:
        tuple: Test data and labels.
    """
    shift_h_data = np.load('data/test/shiftH.npy')
    shift_h_data = np.expand_dims(shift_h_data, axis=-1)

    type_h_data = np.load('data/test/typeH.npy')
    type_h_data = np.expand_dims(type_h_data, axis=-1)

    type_c_data = np.load('data/test/typeC.npy')
    type_c_data = np.expand_dims(type_c_data, axis=-1)

    shift_c_data = np.load('data/test/shiftC.npy')
    shift_c_data = np.expand_dims(shift_c_data, axis=-1)

    y_test = np.load('data/test/label.npy')

    return shift_c_data, type_c_data, shift_h_data, type_h_data, y_test


def evaluate_model(model_path, x_test, y_test):
    """
    Loads a model and evaluates its performance on the test data.

    Args:
        model_path (str): Path to the saved model.
        x_test (tuple): Tuple containing the test data inputs.
        y_test (array-like): True labels for the test data.

    Returns:
        dict: Evaluation metrics including confusion matrix, accuracy, F1 score, recall, and precision.
    """
    model = load_model(model_path)
    y_pred = model.predict(x_test)
    y_labels = np.argmax(y_pred, axis=1)

    conf_matrix = confusion_matrix(y_test, y_labels)
    accuracy = accuracy_score(y_test, y_labels)
    f1 = f1_score(y_test, y_labels, average='weighted')
    recall = recall_score(y_test, y_labels, average='weighted')
    precision = precision_score(y_test, y_labels, average='weighted')

    return {
        'confusion_matrix': conf_matrix,
        'accuracy': accuracy,
        'f1_score': f1,
        'recall': recall,
        'precision': precision
    }


def plot_confusion_matrix(cm, save_path='data/matrix.png', title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots and saves the confusion matrix.

    Args:
        cm (array-like): Confusion matrix.
        save_path (str): Path to save the confusion matrix plot.
        title (str): Title of the plot.
        cmap (matplotlib.colors.Colormap): Colormap to be used for the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]:.0f}', horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', transparent=False)
    plt.show()


def main():
    """
    Main function to execute the model evaluation pipeline.
    """
    configure_gpu(gpu_index=1)

    # Load test data
    shift_c_data, type_c_data, shift_h_data, type_h_data, y_test = load_test_data()

    # Evaluate the model
    evaluation_metrics = evaluate_model(
        model_path='data/modelpath/test4/model_009.hdf5',
        x_test=[shift_c_data, type_c_data, shift_h_data, type_h_data],
        y_test=y_test
    )

    # Print evaluation metrics
    print('confusion_matrix:')
    print(evaluation_metrics['confusion_matrix'])
    print(f"acc: {evaluation_metrics['accuracy']:.4f}")
    print(f"F1: {evaluation_metrics['f1_score']:.4f}")
    print(f"recall: {evaluation_metrics['recall']:.4f}")
    print(f"precision: {evaluation_metrics['precision']:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(evaluation_metrics['confusion_matrix'])


if __name__ == '__main__':
    main()
