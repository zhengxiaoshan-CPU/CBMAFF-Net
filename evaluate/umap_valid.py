import numpy as np
import tensorflow as tf
from umap import UMAP
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def load_data():
    """
    Loads and preprocesses the test data for UMAP visualization.

    Returns:
        tuple: Preprocessed data (shift_c_data, type_c_data, shift_h_data, type_h_data) and labels (Y_test).
    """
    shift_h_data = np.load('data/simul_umap/shiftH.npy')
    shift_h_data = np.expand_dims(shift_h_data, axis=-1)

    type_h_data = np.load('data/simul_umap/typeH.npy')
    type_h_data = np.expand_dims(type_h_data, axis=-1)

    type_c_data = np.load('data/simul_umap/typeC.npy')
    type_c_data = np.expand_dims(type_c_data, axis=-1)

    shift_c_data = np.load('data/simul_umap/shiftC.npy')
    shift_c_data = np.expand_dims(shift_c_data, axis=-1)

    y_test = np.load('data/simul_umap/label.npy', allow_pickle=True)

    return shift_c_data, type_c_data, shift_h_data, type_h_data, y_test


def extract_features(model_path, x_test):
    """
    Extracts features from the last convolutional layer of a pre-trained model.

    Args:
        model_path (str): Path to the pre-trained model.
        x_test (tuple): Tuple containing the test data inputs.

    Returns:
        np.ndarray: Extracted features from the last convolutional layer.
    """
    model = load_model(model_path)
    last_conv_layer = model.layers[-2]  # Assuming the last convolutional layer is the second to last layer
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=last_conv_layer.output)
    features = feature_extractor.predict(x_test)
    return features


def apply_umap(features, n_neighbors=500, min_dist=1, n_components=3, random_state=42):
    """
    Applies UMAP to reduce the dimensionality of the extracted features.

    Args:
        features (np.ndarray): Features extracted from the CNN model.
        n_neighbors (int): Number of neighbors considered for UMAP.
        min_dist (float): Minimum distance between points in the low-dimensional space.
        n_components (int): Number of dimensions to reduce to.
        random_state (int): Seed for reproducibility.

    Returns:
        np.ndarray: UMAP embeddings of the features.
    """
    reducer = UMAP(
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        n_components=n_components, 
        random_state=random_state
    )
    embedding = reducer.fit_transform(features)
    return embedding


def plot_umap(embedding, save_path='data/matrix4.png'):
    """
    Plots the UMAP embeddings and saves the plot as an image.

    Args:
        embedding (np.ndarray): UMAP embeddings of the features.
        save_path (str): Path to save the UMAP plot.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.figure(figsize=(14, 6))

    # Scatter plot for each class
    plt.scatter(embedding[:1715, 0], embedding[:1715, 1], c='red', s=5, label='negative')
    plt.scatter(embedding[1715:2335, 0], embedding[1715:2335, 1], c='blue', s=5, label='synthetic cannabinoids')
    plt.scatter(embedding[2335:2830, 0], embedding[2335:2830, 1], c='green', s=5, label='synthetic cathinones')
    plt.scatter(embedding[2830:3080, 0], embedding[2830:3080, 1], c='purple', s=5, label='phenethylamines')
    plt.scatter(embedding[3080:3625, 0], embedding[3080:3625, 1], c='orange', s=5, label='fentanyl')
    plt.scatter(embedding[3625:3950, 0], embedding[3625:3950, 1], c='maroon', s=5, label='tryptamines')

    # Set labels and legend
    plt.xlabel('UMAP_1', fontdict={'family': 'Times New Roman', 'weight': 'bold'})
    plt.ylabel('UMAP_2', fontdict={'family': 'Times New Roman', 'weight': 'bold'})
    plt.legend()

    # Save the plot
    plt.savefig(save_path, dpi=200, bbox_inches='tight', transparent=False)
    plt.show()


def main():
    """
    Main function to execute the UMAP visualization pipeline.
    """
    # Load and preprocess data
    shift_c_data, type_c_data, shift_h_data, type_h_data, y_test = load_data()

    # Extract features using a pre-trained model
    features = extract_features('data/model_009.hdf5', 
                                [shift_c_data, type_c_data, shift_h_data, type_h_data])

    # Apply UMAP for dimensionality reduction
    embedding = apply_umap(features)

    # Plot and save the UMAP embeddings
    plot_umap(embedding)


if __name__ == '__main__':
    main()
