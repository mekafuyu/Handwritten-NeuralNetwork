import os
import shutil
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt

dict_letras = {
  '001': '0',
  '002': '1',
  '003': '2',
  '004': '3',
  '005': '4',
  '006': '5',
  '007': '6',
  '008': '7',
  '009': '8',
  '010': '9',
  '011': 'ua',
  '012': 'ub',
  '013': 'uc',
  '014': 'ud',
  '015': 'ue',
  '016': 'uf',
  '017': 'ug',
  '018': 'uh',
  '019': 'ui',
  '020': 'uj',
  '021': 'uk',
  '022': 'ul',
  '023': 'um',
  '024': 'un',
  '025': 'uo',
  '026': 'up',
  '027': 'uq',
  '028': 'ur',
  '029': 'us',
  '030': 'ut',
  '031': 'uu',
  '032': 'uv',
  '033': 'uw',
  '034': 'ux',
  '035': 'uy',
  '036': 'uz',
  '037': 'la',
  '038': 'lb',
  '039': 'lc',
  '040': 'ld',
  '041': 'le',
  '042': 'lf',
  '043': 'lg',
  '044': 'lh',
  '045': 'li',
  '046': 'lj',
  '047': 'lk',
  '048': 'll',
  '049': 'lm',
  '050': 'ln',
  '051': 'lo',
  '052': 'lp',
  '053': 'lq',
  '054': 'lr',
  '055': 'ls',
  '056': 'lt',
  '057': 'lu',
  '058': 'lv',
  '059': 'lw',
  '060': 'lx',
  '061': 'ly',
  '062': 'lz'
}

dict_letras_ofc = {
  '001': '0',
  '002': '1',
  '003': '2',
  '004': '3',
  '005': '4',
  '006': '5',
  '007': '6',
  '008': '7',
  '009': '8',
  '010': '9',
  '011': 'a',
  '012': 'b',
  '013': 'c',
  '014': 'd',
  '015': 'e',
  '016': 'f',
  '017': 'g',
  '018': 'h',
  '019': 'i',
  '020': 'j',
  '021': 'k',
  '022': 'l',
  '023': 'm',
  '024': 'n',
  '025': 'o',
  '026': 'p',
  '027': 'q',
  '028': 'r',
  '029': 's',
  '030': 't',
  '031': 'u',
  '032': 'v',
  '033': 'w',
  '034': 'x',
  '035': 'y',
  '036': 'z',
  '037': 'A',
  '038': 'B',
  '039': 'C',
  '040': 'D',
  '041': 'E',
  '042': 'F',
  '043': 'G',
  '044': 'H',
  '045': 'I',
  '046': 'J',
  '047': 'K',
  '048': 'L',
  '049': 'M',
  '050': 'N',
  '051': 'O',
  '052': 'P',
  '053': 'Q',
  '054': 'R',
  '055': 'S',
  '056': 'T',
  '057': 'U',
  '058': 'V',
  '059': 'W',
  '060': 'X',
  '061': 'Y',
  '062': 'Z'
}

# Separar em pastas
def cropResize(path, oX: int, oY: int, nX = -1, nY = -1, crop = True):
  leftOffset = int(oX / 2 - oY / 2)
  for folder in os.listdir(path):
    curr_path = os.path.join(path, folder)
    for file in os.listdir(curr_path):
      if '.' not in file:
        continue
      input = os.path.join(curr_path, file)
      img = cv.imread(input)
      if(crop):
        img = img[0: oY, leftOffset: leftOffset + oY]
      if(nX > 0 and nY > 0):
        img = cv.resize(img, (nX, nY))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, img = cv.threshold(
          img, 127, 255, cv.THRESH_BINARY
        )
      cv.imwrite(input, img)
    
# Separar em pastas
def organize(path):
  for file in os.listdir(path):
    if '.' not in file:
      continue
    
    # Construct full paths
    key = file.split('-')[0][3:]
    destination = os.path.join(path, dict_letras[key])
    if not os.path.exists(destination):
      os.makedirs(destination)
    
    source_path = os.path.join(path, file)
    destination_path = os.path.join(destination, file)
    
    # Move the file
    shutil.move(source_path, destination_path)
    
def transformImage(path, name, func):
  temp = os.path.basename(os.path.normpath(path))
  new_path = os.path.join(os.path.dirname(path), f'{temp}{name}')
  if not os.path.exists(new_path):
    os.mkdir(new_path)
  
  for folder in os.listdir(path):
    curr_path = os.path.join(path, folder)
    curr_new_path = os.path.join(new_path, folder)
    if not os.path.exists(curr_new_path):
      os.mkdir(curr_new_path)
    for file in os.listdir(curr_path):
      if '.' not in file:
        continue
      input = os.path.join(curr_path, file)
      destination_file = os.path.join(curr_new_path, func.__name__ + file)
      img = cv.imread(input)
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      cv.imwrite(destination_file, func(img))

def binarize(input):
  _, img = cv.threshold(
    input, 127, 255, cv.THRESH_BINARY
  )
  return img
 
def fourier(input):
  return mag(fft(input))

def randomizeThickness(img, min=-40, max=40):
  val = random.randrange(min, max, 1)
  if val < 0:
    temp = np.abs(val)
    img = cv.erode(img, np.ones((temp, temp)))
  else:
    img = cv.dilate(img, np.ones((val, val)))
  return img


# Transformada de Fourier
def fft(img):
  img = np.fft.fft2(img)
  img = np.fft.fftshift(img)
  return img
# Obtém a magnitude da imagem
def mag(img):
  absvalue = np.abs(img)
  magnitude = 20 * np.log(absvalue)
  return magnitude

# Inversa (retorna para imagem original)
def ifft(fimg):
  fimg = np.fft.ifftshift(fimg)
  fimg = np.fft.ifft2(fimg)
  return fimg


# Normaliza a imagem entre 0 e 255
def norm(img):
  img = cv.normalize(
    img, None, 0, 255,
    cv.NORM_MINMAX
  )

# Melhor para ver imagens da transformada e imagens pequenas em geral.
def show(img):
  plt.imshow(img, cmap='gray')
  plt.show()
  return img