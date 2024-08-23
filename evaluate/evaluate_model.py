import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score


def load_data(data_path):
    """
    Loads and preprocesses the test data for model evaluation.

    Args:
        data_path (str): The base path where the test data is stored.

    Returns:
        tuple: Preprocessed data (shift_c_data, type_c_data, shift_h_data, type_h_data, y_test).
    """
    shift_h_data = np.load(f'{data_path}/shiftH.npy')
    shift_h_data = np.expand_dims(shift_h_data, axis=-1)

    type_h_data = np.load(f'{data_path}/typeH.npy')
    type_h_data = np.expand_dims(type_h_data, axis=-1)

    type_c_data = np.load(f'{data_path}/typeC.npy')
    type_c_data = np.expand_dims(type_c_data, axis=-1)

    shift_c_data = np.load(f'{data_path}/shiftC.npy')
    shift_c_data = np.expand_dims(shift_c_data, axis=-1)

    # Determine y_test based on the data path
    if 'validset2' in data_path:
        y_test = [
            4, 4, 4, 3, 4, 4, 5, 5, 2, 4, 4, 1, 4, 2, 4, 5, 5, 5, 3, 2, 4, 4,
            2, 4, 2, 2, 1, 1, 2, 4, 2, 3, 2, 2, 2, 4, 3, 2, 4, 4, 4, 4
        ]
    elif 'validset1' in data_path:
        y_test = [
            2, 4, 5, 2, 3, 2, 1, 3, 1, 5, 1, 5, 3, 1, 1, 1, 1, 5, 5, 5
        ]
    else:
        raise ValueError(f"Unknown data path: {data_path}")

    return shift_c_data, type_c_data, shift_h_data, type_h_data, y_test


def evaluate_model(model_path, x_test, y_test):
    """
    Loads a model and evaluates its performance on the test data.

    Args:
        model_path (str): Path to the saved model.
        x_test (tuple): Tuple containing the test data inputs.
        y_test (list): True labels for the test data.

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


def print_evaluation_metrics(metrics):
    """
    Prints the evaluation metrics including confusion matrix, accuracy, F1 score, recall, and precision.

    Args:
        metrics (dict): A dictionary containing evaluation metrics.
    """
    print('confusion_matrix:')
    print(metrics['confusion_matrix'])
    print(f"acc: {metrics['accuracy']:.4f}")
    print(f"F1: {metrics['f1_score']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")


def main():
    """
    Main function to load data, evaluate the model, and print the evaluation metrics.
    """
    # Define the data path (change to 'data/validset1' or 'data/validset2' as needed)
    data_path = 'data/validset2'  # Change to 'data/validset1' if needed

    # Load and preprocess data
    shift_c_data, type_c_data, shift_h_data, type_h_data, y_test = load_data(data_path)

    # Evaluate the model
    metrics = evaluate_model(
        model_path='data/model/model_009.hdf5',
        x_test=[shift_c_data, type_c_data, shift_h_data, type_h_data],
        y_test=y_test
    )

    # Print the evaluation metrics
    print_evaluation_metrics(metrics)


if __name__ == '__main__':
    main()
