import cv2 as cv
from datetime import datetime

def segment(im):
    newImage = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    _, newImage = cv.threshold(newImage, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    figures = []

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
                while len(queue) > 0:
                    currX, currY = queue.pop(0)
                    upper, left, bottom, right = expand(newImage, upper, left, bottom, right, currX, currY, queue)
                    
                    # if cv.waitKey(1) & 0xFF == ord('q'):
                    #     break
                figures.append(((upper, left), (bottom, right)))
                # cv.waitKey(0)
                # cv.destroyAllWindows()
    return figures

def expand(im, u, l, b, r, x, y, queue: list):
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
    return u, l, b, r




im = cv.imread('./upload_folder/inquisicao.png')
im = cv.bitwise_not(im)
cv.imshow('result', im)
cv.waitKey(0)
cv.destroyAllWindows()
# im = cv.resize(im, (128, 128))
start = datetime.now()
figures = segment(im)
print((datetime.now() - start).microseconds)
for figure in figures:
    cv.rectangle(im, figure[0], figure[1], (0,255,0), 3)

cv.imshow('result', im)
cv.waitKey(0)
cv.destroyAllWindows()