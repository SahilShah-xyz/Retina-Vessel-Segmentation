import streamlit as st
from PIL import Image
import cv2
import numpy as np
import cv2
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)

from model import build_unet


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask


def calculate_metrics(y_true, y_pred):
    # Ground truth
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def init_unet_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(f"{path}", map_location=device))
    return model


st.title("Retinal Vessel Segmentation")

# Select Model Checkpoint
path_mods = st.selectbox(
    "Select Checkpoint", ["Toy Unet", "Unet 500", "Unet Drive 200"]
)
up = st.file_uploader("Upload Retinal Image", type=["jpg", "png", "tif"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if path_mods == "Toy Unet":
    model = init_unet_model("../models/toy.pth")

elif path_mods == "Unet 500":
    model = init_unet_model("../models/unet_500.pth")

elif path_mods == "Unet Drive 200":
    model = init_unet_model("../models/unet_200_drive.pth")

if up != None:
    z = Image.open(up)
    image = cv2.cvtColor(
        cv2.resize(
            np.array(z),
            (512, 512),
        ),
        cv2.COLOR_RGB2BGR,
    )  ## (512, 512, 3)

    x = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)  ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)

    with torch.no_grad():
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_y_cp = pred_y
        pred_y = pred_y[0].cpu().numpy()  ## (1, 512, 512)
        pred_y = np.squeeze(pred_y, axis=0)  ## (512, 512)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)

        pred_y = mask_parse(pred_y)

    left, right = st.columns(2)

    with left:
        st.markdown("## Input Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    with right:
        st.markdown("## Segmented Vessel")
        st.image(pred_y * 255)

    st.markdown("## Outputs")
    left2, right2 = st.columns(2)
    with right2:
        inv_mask = 1 - (pred_y)
        unmasked = cv2.cvtColor(cv2.multiply(image, inv_mask), cv2.COLOR_BGR2RGB)

        st.image(
            cv2.addWeighted(
                cv2.cvtColor(
                    cv2.cvtColor(unmasked, cv2.COLOR_BGR2GRAY), cv2.COLOR_BGR2RGB
                ),
                1,
                cv2.cvtColor(image * pred_y, cv2.COLOR_BGR2RGB),
                1,
                0.0,
            ),
            caption="Contrast Output",
        )

    with left2:
        st.image(
            cv2.addWeighted(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                0.3,
                cv2.cvtColor(image * pred_y, cv2.COLOR_BGR2RGB),
                1,
                1,
            ),
            caption="Overlay Output",
        )
    st.markdown("## Check Accuracy")
    up_acc = st.file_uploader("Upload Retinal Mask", type=["jpg", "png"])

    if up_acc != None:
        t = Image.open(up_acc)
        y = cv2.resize(
            np.array(t),
            (512, 512),
        )  ## (512, 512, 3)
        try:
            mask = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)  ## (512, 512)
        except:
            mask = y
        mask = cv2.resize(mask, (512, 512))

        y = np.expand_dims(mask, axis=0)  ## (1, 512, 512)
        y = y / 255.0
        y = np.expand_dims(y, axis=0)  ## (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)
        score = calculate_metrics(y, pred_y_cp)

        left3, right3 = st.columns(2)
        with left3:
            st.image(t, caption="Original Mask")
        with right3:
            st.image(pred_y * 255, caption="Generated Mask")

        st.write(f"Pixel Accuracy: {score[-1] * 100:2f}%")
        st.write(f"Jaccard Similarity: {score[2] * 100:2f}%")
