from genericpath import exists
import os
import requests
from tqdm import tqdm

URL_REPO = "https://github.com/azizhamza-code/NLP/"


def download_file(url, file_path):

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length'))

    try:
        with open(file_path, 'wb', buffering=16*1024*1024) as f:
            bar = tqdm(total=total_size, unit='B', unit_scale=True)
            bar.set_description(os.path.split(file_path)[-1])
            for chunk in r.iter_content(32*1024):
                f.write(chunk)
                bar.update(len(chunk))
            bar.close()
    except Exception:
        print("download failed")

    finally:
        if os.path.getsize(file_path) != total_size:
            os.remove(file_path)
            print("Removed incomplete download")


def download_from_github(version, fn, target_dir, force=False):

    url = URL_REPO + f"releases/download/{version}/{fn}"
    file_path = os.path.join(target_dir, fn)
    if os.path.exists(file_path) and not force:
        print(f"File{file_path} is already download")
        return

    download_file(url, file_path)


def download_iter(target_dir, fns, version, force=False):
    os.makedirs(target_dir, exist_ok=True)
    for fn in fns:
        download_from_github(version, fn, target_dir, force)


def data_bag_of_words(target_dir="data", force=False):

    download_iter(target_dir=target_dir,
                  fns=["train.tsv", "validation.tsv",
                       "test.tsv", "text_prepare_tests.tsv"],
                  version="datav1",
                  force=False

                  )
