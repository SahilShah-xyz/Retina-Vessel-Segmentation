import os
import numpy as np
from rich.progress import track
import shutil


def train_val_split(ratio, src, dst):
    dir_img = np.array(os.listdir(f"{src}img"))
    dir_vessel = np.array(os.listdir(f"{src}vessel"))
    rand_seq = np.random.choice(range(0, len(dir_img)), len(dir_img), replace=False)

    for c, i in enumerate(track(rand_seq)):
        print((c * ratio) // 1, i)
        if c <= (len(dir_img) * ratio) // 1:
            shutil.copyfile(f"{src}img/{dir_img[i]}", f"{dst}test/img/{dir_img[i]}")
            shutil.copyfile(
                f"{src}vessel/{dir_vessel[i]}", f"{dst}test/vessel/{dir_vessel[i]}"
            )
        else:
            shutil.copyfile(f"{src}img/{dir_img[i]}", f"{dst}train/img/{dir_img[i]}")
            shutil.copyfile(
                f"{src}vessel/{dir_vessel[i]}", f"{dst}train/vessel/{dir_vessel[i]}"
            )
    print(f"Train size: {len(np.array(os.listdir(f'{dst}train/img/')))}")
    print(f"Test size: {len(np.array(os.listdir(f'{dst}test/img/')))}")


def move_rand(count, src, dst):
    dir_img = np.array(os.listdir(f"{src}img"))
    dir_vessel = np.array(os.listdir(f"{src}vessel"))
    rand_seq = np.random.choice(range(0, len(dir_img)), count, replace=False)

    for c, i in enumerate(track(rand_seq)):
        shutil.copyfile(f"{src}img/{dir_img[i]}", f"{dst}img/{dir_img[i]}")
        shutil.copyfile(f"{src}vessel/{dir_vessel[i]}", f"{dst}vessel/{dir_vessel[i]}")


if __name__ == "__main__":
    train_val_split(0.3, "../data/smol_split/mix/", "../data/smol_split/")
    # move_rand(100, "../data/clean_mix/", "../data/smol_split/mix/")
