import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from umap import UMAP


class Fingerprint:
    """
    A class to represent a molecular fingerprint.

    Attributes:
        fp (np.ndarray): The fingerprint as a numpy array.
        names (list): The names of the bits in the fingerprint.
    """
    
    def __init__(self, fp, names):
        self.fp = fp
        self.names = names

    def __str__(self):
        return f"{len(self.fp)} bit FP"

    def __len__(self):
        return len(self.fp)


def get_circular_fingerprint(mol, radius=2, nBits=1024, useFeatures=False, counts=False, dtype=np.float32):
    """
    Generates a circular (Morgan) fingerprint for a molecule.

    Args:
        mol (rdkit.Chem.Mol): The molecule for which to generate the fingerprint.
        radius (int): The radius for the Morgan fingerprint.
        nBits (int): The size of the fingerprint.
        useFeatures (bool): Whether to use feature invariants.
        counts (bool): Whether to count the number of occurrences of each bit.
        dtype (type): The data type for the fingerprint.

    Returns:
        Fingerprint: The generated fingerprint as a Fingerprint object.
    """
    arr = np.zeros((1,), dtype)
    if counts:
        info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures, bitInfo=info)
        arr = np.array([len(info[x]) if x in info else 0 for x in range(nBits)], dtype)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures)
        DataStructs.ConvertToNumpyArray(fp, arr)
    
    return Fingerprint(arr, range(nBits))


def calculate_fingerprint(smi):
    """
    Calculates the fingerprint for a given SMILES string.

    Args:
        smi (str): SMILES string of the molecule.

    Returns:
        Fingerprint: The fingerprint of the molecule, or None if calculation fails.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        return get_circular_fingerprint(mol)
    except Exception as e:
        print(f"Error processing SMILES {smi}: {e}")
        return None


def generate_umap_embeddings(fp_list, n_components=3, n_neighbors=500, min_dist=1, random_state=42):
    """
    Generates UMAP embeddings from a list of fingerprints.

    Args:
        fp_list (list of Fingerprint): The list of fingerprints.
        n_components (int): The number of dimensions to reduce to.
        n_neighbors (int): The number of neighbors for UMAP.
        min_dist (float): The minimum distance between points in the low-dimensional space.
        random_state (int): Random seed for reproducibility.

    Returns:
        np.ndarray: UMAP embeddings of the fingerprints.
    """
    X = np.array([fp.fp for fp in fp_list])
    umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return umap_model.fit_transform(X)


def plot_umap(df, output_path='data/pic/umap.png'):
    """
    Plots UMAP embeddings and saves the plot as an image.

    Args:
        df (pd.DataFrame): DataFrame containing the UMAP coordinates and labels.
        output_path (str): Path to save the plot image.
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Support for Chinese characters
    plt.rcParams['axes.unicode_minus'] = False    # Support for negative signs
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams['font.weight'] = 'bold'

    fig, ax = plt.figure(figsize=(8, 6)), plt.gca()

    # Plot each cluster with a different color
    ax.scatter(df['UMAP_1'].iloc[1:345], df['UMAP_2'].iloc[1:345], s=5, alpha=0.5, label='negative')
    ax.scatter(df['UMAP_1'].iloc[345:464], df['UMAP_2'].iloc[345:464], s=5, alpha=0.5, label='synthetic cannabinoids')
    ax.scatter(df['UMAP_1'].iloc[464:554], df['UMAP_2'].iloc[464:554], s=5, alpha=0.5, label='synthetic cathinones')
    ax.scatter(df['UMAP_1'].iloc[554:598], df['UMAP_2'].iloc[554:598], s=5, alpha=0.5, label='phenethylamines')
    ax.scatter(df['UMAP_1'].iloc[598:707], df['UMAP_2'].iloc[598:707], s=5, alpha=0.5, label='fentanyl')
    ax.scatter(df['UMAP_1'].iloc[707:771], df['UMAP_2'].iloc[707:771], s=5, alpha=0.5, label='tryptamines')

    font = {'family': 'Times New Roman', 'weight': 'bold'}
    ax.set_xlabel('UMAP_1', fontdict=font)
    ax.set_ylabel('UMAP_2', fontdict=font)
    ax.legend()

    plt.savefig(output_path, dpi=200)
    plt.show()


def main():
    """
    Main function to perform UMAP embedding and visualization.
    """
    # Load the dataset
    df = pd.read_csv('data/dataset_smiles.csv')
    
    # Calculate fingerprints
    df['FP'] = df['smiles'].apply(calculate_fingerprint)
    print("First 3 rows of the dataset with fingerprints:")
    print(df.head(3))

    # Save intermediate data
    df.to_csv('data/hhhhh.csv', sep=',', index=False, header=False)

    # Generate UMAP embeddings
    fp_list = [fp for fp in df['FP'] if fp is not None]
    embeddings = generate_umap_embeddings(fp_list)

    # Add UMAP embeddings to DataFrame
    df = df[df['FP'].notna()]
    df['UMAP_1'], df['UMAP_2'] = embeddings[:, 0], embeddings[:, 1]

    print("First 3 rows of the dataset with UMAP embeddings:")
    print(df.head(3))

    # Plot UMAP embeddings
    plot_umap(df)


if __name__ == '__main__':
    main()
