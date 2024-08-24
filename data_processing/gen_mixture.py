import pandas as pd
import random
import argparse

def process_data(
    input_file_path, 
    sheet_name_C, 
    sheet_name_H, 
    excluded_numbers, 
    iterations, 
    fraction=0.1, 
    output_file_path=None
):
    """
    Combines supplementary data from two sheets and then removes a random fraction of rows from each group.
    
    Args:
        input_file_path (str): Path to the input Excel file.
        sheet_name_C (str): Sheet name for C data.
        sheet_name_H (str): Sheet name for H data.
        excluded_numbers (list): List of numbers to exclude from selection.
        iterations (int): Number of iterations for combining data.
        fraction (float): Fraction of rows to remove from each group.
        output_file_path (str): Path to save the output Excel file. If None, will overwrite the input file.
    """
    df_C = pd.read_excel(input_file_path, sheet_name=sheet_name_C)
    df_H = pd.read_excel(input_file_path, sheet_name=sheet_name_H)
    result_df_C = pd.DataFrame()
    result_df_H = pd.DataFrame()
    
    # Combine data
    for i in range(iterations):
        random_number = random.choice([x for x in range(1, 42) if x not in excluded_numbers])
        selected_rows_C = df_C[df_C['ID'] == random_number]
        selected_rows_C['ID'] = f'0{i+1:03d}'
        selected_rows_H = df_H[df_H['ID'] == random_number]
        selected_rows_H['ID'] = f'0{i+1:03d}'
        result_df_C = pd.concat([result_df_C, selected_rows_C])
        result_df_H = pd.concat([result_df_H, selected_rows_H])
    
    # Remove random rows
    final_df_C = pd.DataFrame()
    final_df_H = pd.DataFrame()

    for _, group_df in result_df_C.groupby('ID'):
        num_rows = len(group_df)
        num_rows_to_delete = int(num_rows * fraction)
        if num_rows_to_delete > 0:
            rows_to_delete = random.sample(range(num_rows), num_rows_to_delete)
            remaining_group_df = group_df.drop(group_df.index[rows_to_delete])
            final_df_C = pd.concat([final_df_C, remaining_group_df])
        else:
            final_df_C = pd.concat([final_df_C, group_df])

    for _, group_df in result_df_H.groupby('ID'):
        num_rows = len(group_df)
        num_rows_to_delete = int(num_rows * fraction)
        if num_rows_to_delete > 0:
            rows_to_delete = random.sample(range(num_rows), num_rows_to_delete)
            remaining_group_df = group_df.drop(group_df.index[rows_to_delete])
            final_df_H = pd.concat([final_df_H, remaining_group_df])
        else:
            final_df_H = pd.concat([final_df_H, group_df])

    # Save results to Excel
    output_file_path = output_file_path or input_file_path
    with pd.ExcelWriter(output_file_path) as writer:
        final_df_C.to_excel(writer, sheet_name=sheet_name_C, index=False)
        final_df_H.to_excel(writer, sheet_name=sheet_name_H, index=False)

def main():
    parser = argparse.ArgumentParser(description='Process and combine data from Excel sheets.')
    parser.add_argument('--input_file_path', type=str, required=True, help='Path to the input Excel file.')
    parser.add_argument('--sheet_name_C', type=str, required=True, help='Sheet name for C data.')
    parser.add_argument('--sheet_name_H', type=str, required=True, help='Sheet name for H data.')
    parser.add_argument('--excluded_numbers', type=int, nargs='+', required=True, help='List of numbers to exclude from selection.')
    parser.add_argument('--iterations', type=int, required=True, help='Number of iterations for combining data.')
    parser.add_argument('--fraction', type=float, default=0.1, help='Fraction of rows to remove from each group.')
    parser.add_argument('--output_file_path', type=str, help='Path to save the output Excel file.')

    args = parser.parse_args()

    process_data(
        input_file_path=args.input_file_path,
        sheet_name_C=args.sheet_name_C,
        sheet_name_H=args.sheet_name_H,
        excluded_numbers=args.excluded_numbers,
        iterations=args.iterations,
        fraction=args.fraction,
        output_file_path=args.output_file_path
    )

if __name__ == '__main__':
    main()
