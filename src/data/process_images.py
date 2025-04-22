import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image


def main(answer_file):
    answers = pd.read_csv(answer_file, index_col=0)
    
    for fullpath in tqdm(answers.index):
        new_path = fullpath[:fullpath.rfind('.')] + '.png'

        image = Image.open(fullpath)
        image = image.convert('RGB')
        image = image.crop((5, 25, image.size[0] - 5, image.size[1] - 25))

        width, height = image.size
        if width < 224 or height < 224:
            print(f'Be careful, size of {new_path} is less, then 224')

        Path(fullpath).unlink()
        image.save(new_path)

        if fullpath != new_path:
            answers.loc[new_path] = answers.loc[fullpath]
            answers.drop(fullpath, inplace=True)

        answers.to_csv(answer_file)


if __name__ == '__main__':
    main(*sys.argv[1:])