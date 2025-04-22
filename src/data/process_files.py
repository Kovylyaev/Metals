import sys
import shutil
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from PIL import Image


def main(clear_dir, source_dir, target_dir, file_format, target_value, answer_file):
    if int(clear_dir):
        try:
            shutil.rmtree(target_dir)
        except:
            pass

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_img_dir = Path(target_dir).joinpath('images')
    target_dir.mkdir(parents=True, exist_ok=True)
    target_img_dir.mkdir(parents=True, exist_ok=True)

    if Path(answer_file).exists():
        path_to_C = pd.read_csv(answer_file, index_col=0).T.to_dict()
    else:
        path_to_C = dict()
    file_paths = source_dir.rglob('*.' + file_format)
    
    for file_path in tqdm(file_paths):
        file = file_path.relative_to(source_dir)
        new_file = str(file).replace('/', '_')
        target_path = target_img_dir.joinpath(new_file)
        shutil.copy(file_path, target_path)
        
        path_to_C[target_path] = {'C': target_value}

    answers = pd.DataFrame.from_dict(path_to_C, orient='index')
    answers.to_csv(answer_file)


if __name__ == '__main__':
    main(*sys.argv[1:])