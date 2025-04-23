import sys
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToPILImage


def main(img_dir, props_path, file_format):
    props = pd.read_csv(props_path, index_col=0)
    img_dir = Path(img_dir)
    to_pil = ToPILImage()

    file_paths = img_dir.rglob('*.' + file_format)
    for file_path in file_paths:
        file = file_path.relative_to(file_path.parent)
        new_file = Path(str(file).replace('Cropped', ''))
        new_file_path = Path(str(file_path).replace('Cropped', ''))

        try:
            _ = props.loc[str(new_file)]
        except:
            print(file_path, ' No such row in table')
            Path(file_path).unlink()
            continue
        if pd.isna(props.loc[str(new_file)]['magnification']):
            print(file_path, ' magnification is null')
            Path(file_path).unlink()
            continue
        elif int(props.loc[str(new_file)]['magnification'][:-1]) < 200 or int(props.loc[str(new_file)]['magnification'][:-1]) > 1200:
            print(file_path, ' magnification is too big or too small')
            Path(file_path).unlink()
            continue

        scale_factor = 500 / int(props.loc[str(new_file)]['magnification'][:-1])
        image = Image.open(file_path)
        image = torch.tensor(np.asarray(image.convert('RGB'))).permute(2, 0, 1).unsqueeze(0)
        image_scaled = F.interpolate(image, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        image_scaled = to_pil(image_scaled.squeeze())

        Path(file_path).unlink()
        image_scaled.save(new_file_path)


if __name__ == '__main__':
    main(*sys.argv[1:])