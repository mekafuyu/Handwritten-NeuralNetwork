import os
import cv2 as cv
import numpy as np
from flask import Flask, flash, request, redirect, url_for
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
from keras import models
import utils
import segment

UPLOAD_FOLDER = './upload_folder'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'justfortestingsowhatever'

model = models.load_model("checkpoints/989-995.keras")
test = []
alphabet = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
for i in range(10):
  test.append((str(ord(str(i))), str(i)))
for i in alphabet:
  test.append((str(ord(i)), i))
test.sort()

@app.route("/", methods=["GET", "POST"])
@cross_origin()
def upload_image():
  if request.method == 'POST':
    if 'file' not in request.files:
      flash('No file part')
      return "None"
    if 'nome' not in request.form:
      flash('No fomr part')
      return "None"
    
    nome = request.form.get('nome')
    file = request.files['file']
    
    if file.filename == '':
      flash('No selected file')
      return redirect(request.url)
    
    if nome == '':
      flash('No name')
      return redirect(request.url)
    
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    
    res = []
    letters = cv.imread(path)
    figures = segment.segment(letters)
    for _, _, letter_img in figures:
      im = cv.imread(letter_img)
      im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
      _, im = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
      im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
      im = np.expand_dims(im, axis=0)
      # res.append(utils.dict_letras_ofc[str( + 1).zfill(3)])
      pred = np.argmax(model.predict(im))
      res.append(test[pred][1])
    typed = ''.join(res)
    return {'typed': typed, 'check': typed.lower() == nome.lower()}
  return "a"

