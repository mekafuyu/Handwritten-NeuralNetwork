import cv2 as cv
import numpy as np
from datetime import datetime

def segment(im):
    newImage = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    
    _, newImage = cv.threshold(newImage, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    figures = []
    figures_pixels = []
    for x in range(len(newImage)):
        for y in range(len(newImage[x])):
            if newImage[x][y] == 0:
                currX = x
                currY = y
                upper = im.shape[1]
                bottom = 0
                left = im.shape[0]
                right = 0
                queue = []
                queue.append((x, y))
                figure_pixels = []
                while len(queue) > 0:
                    currX, currY = queue.pop(0)
                    upper, left, bottom, right = expand(newImage, upper, left, bottom, right, currX, currY, queue, figure_pixels)
                    
                figures_pixels.append(figure_pixels)
                figures.append(((upper, left), (bottom, right)))
                
                newfig = np.zeros((right - left + 1, bottom - upper + 1, 1), np.uint8)
                newfig[newfig == 0] = 255

                for x, y in figure_pixels:
                    newfig[x - left][y - upper] = 0
                
                ofcFig = np.zeros((128, 128, 1), np.uint8)
                ofcFig[ofcFig == 0] = 255
                width, heigth, _ = newfig.shape
                
                if heigth > width:
                    width = int(108 / heigth * width)
                    heigth = 108
                    newfig = cv.resize(newfig, (heigth, width))
                    offset = int((128 - width) / 2)
                    for x in range(len(newfig)):
                        for y in range(len(newfig[x])):
                            ofcFig[x + offset][y + 10] = newfig[x][y]
                else:
                    heigth = int(108 / width * heigth)
                    width = 108
                    newfig = cv.resize(newfig, (heigth, width))
                    offset = int((128 - heigth) / 2)
                    for x in range(len(newfig)):
                        for y in range(len(newfig[x])):
                            ofcFig[x + 10][y + offset] = newfig[x][y]
                    
                # cv.destroyAllWindows()
                cv.imwrite('upload_folder/teste{0}.png'.format(len(figures)), ofcFig)

    return figures

def expand(im, u, l, b, r, x, y, queue: list, pixels: list):
    if im[x][y] == 0:
        if x < l:
            l = x
        if x > r:
            r = x
        if y < u:
            u = y
        if y > b:
            b = y
        queue.append((x - 1, y))
        queue.append((x + 1, y))
        queue.append((x, y - 1))
        queue.append((x, y + 1))
        im[x][y] = 128
        pixels.append((x, y))
    return u, l, b, r




im = cv.imread('./upload_folder/teste.png')
# im = cv.bitwise_not(im)
cv.imshow('result', im)
cv.waitKey(0)
cv.destroyAllWindows()
# im = cv.resize(im, (128, 128))
start = datetime.now()
figures = segment(im)
print((datetime.now() - start).microseconds)

def sort_func(e):
    return(e[0][0])

figures.sort(key=sort_func)

for i, figure in enumerate(figures):
    x = int((figure[1][0] - figure[0][0]) / 2 + figure[0][0])
    y = int((figure[1][1] - figure[0][1]) / 2 + figure[0][1])
    image = cv.putText(
        im,
        str(i),
        (x, y),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        1,
        cv.LINE_AA) 
    cv.rectangle(im, figure[0], figure[1], (255 / len(figures) * i, 255 / len(figures) * i, 255), 3)

cv.imshow('result', im)
cv.waitKey(0)
cv.destroyAllWindows()