import sys
import shutil
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np

def weight(data_table, target_column, num_bins=20):
    values = data_table[target_column].values
    bins = np.linspace(values.min(), values.max(), num=num_bins)
    bin_indices = np.digitize(values, bins)
    bin_counts = np.bincount(bin_indices)
    bin_weights = 1.0 / (bin_counts + 1e-6)
    bin_weights = bin_weights * (len(values) / bin_weights[bin_indices].sum())
    weights = bin_weights[bin_indices]
    data_table['weight'] = weights
    return data_table


def main(clear_dir, source_dir, target_dir, file_format, source_answer_file, target_column, target_answer_file):
    if int(clear_dir):
        try:
            shutil.rmtree(target_dir)
        except:
            raise ValueError(f"Can't delete directory {target_dir}")

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_img_dir = Path(target_dir).joinpath('images')
    target_dir.mkdir(parents=True, exist_ok=True)
    target_img_dir.mkdir(parents=True, exist_ok=True)

    source_answer_file = pd.read_csv(source_answer_file, index_col=1)
    if Path(target_answer_file).exists():
        path_to_C = pd.read_csv(target_answer_file, index_col=0).T.to_dict()
    else:
        path_to_C = dict()
    file_paths = source_dir.rglob('*.' + file_format)
    
    for file_path in tqdm(file_paths):
        file = file_path.relative_to(source_dir)
        new_file = str(file).replace('/', '_')
        subdir = file.parent
        target_path = target_img_dir.joinpath(new_file)
        
        target_value = source_answer_file.loc[str(subdir)].iloc[0][target_column]
        if pd.isna(target_value):
            continue
        path_to_C[target_path.resolve()] = {target_column: target_value}

        shutil.copy(file_path, target_path)

    answers = pd.DataFrame.from_dict(path_to_C, orient='index')
    answers = weight(answers, target_column)
    answers.to_csv(target_answer_file)


if __name__ == '__main__':
    main(*sys.argv[1:])