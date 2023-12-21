import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm


def generate(src, dst, type):
    dir = np.array(os.listdir(src))
    df = {"Name": [], "Origin": [], "Type": type, "Width": [], "Height": [], "Path": []}

    for i in tqdm(dir):
        r_pth = src + i
        img = cv2.imread(r_pth)
        height, width, ch = img.shape
        _ = i.split("_")
        df["Name"].append("_".join(_[1:]))
        df["Origin"].append(_[0])
        df["Width"].append(width)
        df["Height"].append(height)
        df["Path"].append(r_pth)

    exp_fr = pd.DataFrame(df)
    exp_fr.to_csv(dst, index=True, index_label="IDX")


generate(
    "../data/cleaned/test/img/",
    "../data_csv/cleaned_csv/test_img.csv",
    "RGB_Img",
)
