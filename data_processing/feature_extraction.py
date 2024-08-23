# feature_extraction.py

import pandas as pd
import numpy as np


def data_encoder_C(df):
    """
    Encodes C shift data into a one-dimensional binary vector.

    Args:
        df (pd.DataFrame): DataFrame containing the C shift data.

    Returns:
        pd.DataFrame: Encoded DataFrame with binary values indicating shifts.
    """
    min_shift = 0
    max_shift = 220
    new_line_length = (max_shift - min_shift) * 10
    new_line = [0] * new_line_length

    new_data = []
    for _, row in df.iterrows():
        new_line_copy = new_line.copy()
        for shift in row:
            if min_shift <= shift <= max_shift:
                index = int((shift - min_shift) * 10) - 1
                if 0 <= index < new_line_length:
                    new_line_copy[index] = 1
        new_data.append(new_line_copy)

    return pd.DataFrame(new_data)


def data_encoder_H(df, df1):
    """
    Encodes H shift data and the corresponding number of H atoms into a one-dimensional vector.

    Args:
        df (pd.DataFrame): DataFrame containing the H shift data.
        df1 (pd.DataFrame): DataFrame containing the number of H atoms.

    Returns:
        pd.DataFrame: Encoded DataFrame with values indicating H atom numbers.
    """
    min_shift = 0
    max_shift = 17
    new_line_length = (max_shift - min_shift) * 100
    new_line = [0] * new_line_length

    new_data = []
    for row_index, row in df.iterrows():
        new_line_copy = new_line.copy()
        for col_name, shift in row.items():
            if min_shift <= shift <= max_shift:
                index = int((shift - min_shift) * 100) - 1
                if 0 <= index < new_line_length:
                    new_line_copy[index] = df1.loc[row_index, col_name]
        new_data.append(new_line_copy)

    return pd.DataFrame(new_data)


def extract_C_shift_data(df_C, ids, output_file_path):
    """
    Extracts and encodes C shift data and saves it as a .npy file.

    Args:
        df_C (pd.DataFrame): DataFrame containing C shift data.
        ids (array-like): Unique IDs to group the data.
        output_file_path (str): Path to save the encoded .npy file.
    """
    grouped_data = df_C.groupby('ID')['δC'].apply(lambda x: pd.Series(x.values)).unstack().reset_index()
    grouped_data = grouped_data.sort_values(by=['ID'], key=lambda x: x.map({id: i for i, id in enumerate(ids)}))
    data_matrix = data_encoder_C(grouped_data.drop(columns=['ID']))
    
    np.save(output_file_path, data_matrix)
    print(f"C shift has been saved：{output_file_path}")


def extract_C_type_data(df_C, ids, output_file_path, max_sequence_length=78):
    """
    Extracts and encodes C type data and saves it as a .npy file.

    Args:
        df_C (pd.DataFrame): DataFrame containing C type data.
        ids (array-like): Unique IDs to group the data.
        output_file_path (str): Path to save the encoded .npy file.
        max_sequence_length (int): Maximum length of the sequence after padding.
    """
    grouped_data = df_C.groupby('ID')[['C_1', 'C_2', 'C_3', 'C_4']].apply(lambda x: x.values.tolist()).reset_index(name='data')
    grouped_data = grouped_data.sort_values(by=['ID'], key=lambda x: x.map({id: i for i, id in enumerate(ids)}))

    data_matrix = np.array([
        np.pad(seq, ((0, max_sequence_length - len(seq)), (0, 0)), 'constant') 
        for seq in grouped_data['data']
    ])
    
    np.save(output_file_path, data_matrix)
    print(f"C type has been saved：{output_file_path}")


def extract_H_shift_and_atom_data(df_H, ids, output_file_path):
    """
    Extracts and encodes H shift data and the corresponding number of H atoms,
    then saves it as a .npy file.

    Args:
        df_H (pd.DataFrame): DataFrame containing H shift data.
        ids (array-like): Unique IDs to group the data.
        output_file_path (str): Path to save the encoded .npy file.
    """
    grouped_data = df_H.groupby('ID')['δH'].apply(lambda x: pd.Series(x.values)).unstack().reset_index()
    grouped_data1 = df_H.groupby('ID')['H原子个数'].apply(lambda x: pd.Series(x.values)).unstack().reset_index()
    grouped_data = grouped_data.sort_values(by=['ID'], key=lambda x: x.map({id: i for i, id in enumerate(ids)}))
    grouped_data1 = grouped_data1.sort_values(by=['ID'], key=lambda x: x.map({id: i for i, id in enumerate(ids)}))
    
    data_matrix = data_encoder_H(grouped_data.drop(columns=['ID']), grouped_data1.drop(columns=['ID']))
    
    np.save(output_file_path, data_matrix)
    print(f"H shift and H num have been saved：{output_file_path}")


def extract_H_type_data(df_H, ids, output_file_path, max_sequence_length=71):
    """
    Extracts and encodes H type data and saves it as a .npy file.

    Args:
        df_H (pd.DataFrame): DataFrame containing H type data.
        ids (array-like): Unique IDs to group the data.
        output_file_path (str): Path to save the encoded .npy file.
        max_sequence_length (int): Maximum length of the sequence after padding.
    """
    grouped_data = df_H.groupby('ID')[['H_0', 'H_1', 'H_2', 'H_3', 'H_4', 'H_5', 'H_6', 'H_7', 'H_8']].apply(lambda x: x.values.tolist()).reset_index(name='data')
    grouped_data = grouped_data.sort_values(by=['ID'], key=lambda x: x.map({id: i for i, id in enumerate(ids)}))

    data_matrix = np.array([
        np.pad(seq, ((0, max_sequence_length - len(seq)), (0, 0)), 'constant') 
        for seq in grouped_data['data']
    ])
    
    np.save(output_file_path, data_matrix)
    print(f"H type has been saved：{output_file_path}")


def extract_labels(df_H, ids, output_file_path):
    """
    Extracts and saves label data as a .npy file.

    Args:
        df_H (pd.DataFrame): DataFrame containing label data.
        ids (array-like): Unique IDs to group the data.
        output_file_path (str): Path to save the labels .npy file.
    """
    grouped_labels = df_H.groupby('ID')['label'].first().reset_index()
    grouped_labels = grouped_labels.sort_values(by=['ID'], key=lambda x: x.map({id: i for i, id in enumerate(ids)}))
    
    np.save(output_file_path, grouped_labels['label'].values)
    print(f"label has been saved：{output_file_path}")


def main():
    """
    Main function to execute the feature extraction pipeline.
    """
    df_H = pd.read_excel('data/test_data.xlsx', 'H')
    df_C = pd.read_excel('data/test_data.xlsx', 'C')
    ids = df_H['ID'].unique()

    extract_C_shift_data(df_C, ids, output_file_path='data/test/shiftC.npy')
    extract_C_type_data(df_C, ids, output_file_path='data/test/typeC.npy')
    extract_H_shift_and_atom_data(df_H, ids, output_file_path='data/test/shiftH.npy')
    extract_H_type_data(df_H, ids, output_file_path='data/test/typeH.npy')
    extract_labels(df_H, ids, output_file_path='data/test/label.npy')


if __name__ == '__main__':
    main()
