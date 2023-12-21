# Retinal Vessel Segmentation

This project aims at establishing a training and modelling piepline for **U-Net** based Deep Encoder models. It uses the DRIVE dataset composed of Retinal Tomography scans for model training in order to produce a segmentation model capable of identifying and producing a mask of the intricate retinal vessels.

**Research Paper describing the project available in './project' directory**


## User Interface

![UI](./img/ui.png 'UI')
Implemented using Streamlit

## Network Architecture

![Unet](./img/unet.png 'UNET')
Based on the Published Paper: https://arxiv.org/abs/1505.04597

## Dataset Used

![DRIVE](./img/drive.png 'DRIVE')
DRIVE Dataset: https://drive.grand-challenge.org/

## Results

![results](./img/results.png 'results')

## Train Summary

Due to the immense computation required to train U-Net models, I trained a hypothesis toy model which was trained over a single image and a generalized model trained over the full dataset. The model does converge however more data is required to get a good fitting model. Implementation of residual blocks or attention blocks is possible, however such models will require high computation as well as time to converge optimally.