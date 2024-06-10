import sys
from zipfile import ZipFile
from pathlib import Path
from PIL import Image


target_dir = Path('raw_image')


with ZipFile(sys.argv[1]) as zip:
    for path in zip.namelist():
        with zip.open(path) as imf:
            impath = Path(target_dir, path).with_suffix('.png')
            impath.parent.mkdir(parents=True, exist_ok=True)
            Image.open(imf).save(impath)
