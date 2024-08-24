# data_processing/data_processing.py

import pandas as pd
import numpy as np
import random
from .utils import read_excel

def process_peak_types(data_files, sheet_name, column_name):
    peak_types = [read_excel(file, sheet_name, column_name) for file in data_files]
    return pd.unique(pd.concat(peak_types))

def map_peak_types(file_path, sheet_name, column_name, mapping_dict, output_file_path):
    df = pd.read_excel(file_path, sheet_name)
    df['type'] = df[column_name].map(mapping_dict)
    df.to_excel(output_file_path, index=False)

def sort_by_displacement(input_file_path, sheet_name, output_file_path, sort_columns):
    df = pd.read_excel(input_file_path, sheet_name)
    df_sorted = df.sort_values(by=sort_columns)
    with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a') as writer:
        df_sorted.to_excel(writer, sheet_name=sheet_name, index=False)

def one_hot_encode(file_path, sheet_name, column_name, prefix, output_file_path):
    df = pd.read_excel(file_path, sheet_name)
    onehot_encoded = pd.get_dummies(df[column_name], prefix=prefix)
    df = pd.concat([df, onehot_encoded], axis=1)
    with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

def randomize_columns(file_path, sheet_name, output_file_path, columns, fraction=1/3):
    df = pd.read_excel(file_path, sheet_name)
    unique_ids = df['ID'].unique()
    selected_rows = []
    for unique_id in unique_ids:
        id_rows = df[df['ID'] == unique_id]
        n_rows_to_select = int(len(id_rows) * fraction)
        selected_rows.extend(np.random.choice(id_rows.index, size=n_rows_to_select, replace=False))
    df.loc[selected_rows, columns] = 0
    df.to_excel(output_file_path, index=False)

def combine_supplementary_data(input_file_path, sheet_name_C, sheet_name_H, excluded_numbers, iterations, output_file_path):
    df_C = pd.read_excel(input_file_path, sheet_name=sheet_name_C)
    df_H = pd.read_excel(input_file_path, sheet_name=sheet_name_H)
    result_df_C = pd.DataFrame()
    result_df_H = pd.DataFrame()
    for i in range(iterations):
        random_number = random.choice([x for x in range(1, 42) if x not in excluded_numbers])
        selected_rows_C = df_C[df_C['ID'] == random_number]
        selected_rows_C['ID'] = f'0{i+1:03d}'
        selected_rows_H = df_H[df_H['ID'] == random_number]
        selected_rows_H['ID'] = f'0{i+1:03d}'
        result_df_C = pd.concat([result_df_C, selected_rows_C])
        result_df_H = pd.concat([result_df_H, selected_rows_H])
    with pd.ExcelWriter(output_file_path) as writer:
        result_df_C.to_excel(writer, sheet_name=sheet_name_C, index=False)
        result_df_H.to_excel(writer, sheet_name=sheet_name_H, index=False)

def encode_and_extract(input_file_path, sheet_name, group_column, extract_column, output_file_path):
    df = pd.read_excel(input_file_path, sheet_name)
    extracted_list = df.groupby(group_column)[extract_column].apply(list).reset_index(name=f'shift{extract_column}')
    merged_df = pd.merge(df[[group_column]].drop_duplicates(), extracted_list, on=group_column, how='inner')
    merged_df.to_excel(output_file_path, index=False)

def remove_random_rows(file_path, sheet_name_C, sheet_name_H, fraction=0.1):
    df_C = pd.read_excel(file_path, sheet_name=sheet_name_C)
    df_H = pd.read_excel(file_path, sheet_name=sheet_name_H)
    result_df_C = pd.DataFrame()
    result_df_H = pd.DataFrame()
    for _, group_df in df_C.groupby('ID'):
        num_rows = len(group_df)
        num_rows_to_delete = int(num_rows * fraction)
        rows_to_delete = random.sample(range(num_rows), num_rows_to_delete)
        remaining_group_df = group_df.drop(group_df.index[rows_to_delete])
        result_df_C = pd.concat([result_df_C, remaining_group_df])
    for _, group_df in df_H.groupby('ID'):
        num_rows = len(group_df)
        num_rows_to_delete = int(num_rows * fraction)
        rows_to_delete = random.sample(range(num_rows), num_rows_to_delete)
        remaining_group_df = group_df.drop(group_df.index[rows_to_delete])
        result_df_H = pd.concat([result_df_H, remaining_group_df])
    with pd.ExcelWriter(output_file_path) as writer:
        result_df_C.to_excel(writer, sheet_name=sheet_name_C, index=False)
        result_df_H.to_excel(writer, sheet_name=sheet_name_H, index=False)
