import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import model_utils

epochs = 1000
batch_size = 8
patience = 20
learning_rate = 1e-3
model_path = 'checkpoints/'
train_path = './handwritten/Img'
train_set = ''

# organize(train_path)
# transformImage(train_path, '', randomizeThickness)
# cropResize(train_path, 1200, 900, 128, 128)
# transformImage(train_path, '', binarize)
# transformImage(train_path, 'fourier', fourier)
# transformImage(train_path + '-binarize', 'fourier', fourier)

model1 = model_utils.DefaultModel(
  model_path + f'model{train_set}.keras',
  train_path + train_set,
  epochs,
  batch_size,
  patience,
  learning_rate
)
model1.CreateLoadModel()
model1.Split()
model1.Fit()