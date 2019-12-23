import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from time import sleep

class fifo:
    def __init__(self, queue = []):
        self.queue = queue
    def add(self, value):
        self.queue.append(value)
    def remove(self):
        temp = self.queue[0]
        self.queue = self.queue[1:]
        return temp
    def isempty(self):        
        return len(self.queue) == 0

# Minima detection

# Read and grayscale the image
# img = mpimg.imread('./watershed_sample.png')
# img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

# img = np.random.randint(4, size=(8, 8))
arr = [
    [0, 0, 1, 2, 1],
    [0, 0, 2, 1, 0],
    [0, 0, 0, 0, 2],
    [0, 0, 2, 1, 0],
    [1, 3, 1, 1, 0],
]
img = np.array(arr)

INIT = -1
curlab = 1
fifo = fifo()

lab = np.full(img.shape, INIT)

def get_neighbours(matrix, index):
    ret = []
    row, col = index[0], index[1]

    if row - 1 >= 0:
        ret.append((row-1, col))
    if row + 1 < matrix.shape[0]:
        ret.append((row+1, col))
    if col + 1 < matrix.shape[1]:
        ret.append((row, col + 1))
    if col - 1 >= 0:
        ret.append((row, col-1))

    return ret

(rows, cols) = img.shape

pixel_groups = {}
for y in range(rows):
    for x in range(cols):
        if img[(y,x)] in pixel_groups:
            pixel_groups[img[(y,x)]].append((y, x))
        else: 
            pixel_groups[img[(y,x)]] = [(y ,x)]

# print(img)
# print(pixel_groups)

print(img)

for pixel in sorted(pixel_groups.keys()):
    for p in [x for x in pixel_groups[pixel] if lab[x] == INIT]:
        lab[p] = curlab
        fifo.add(p)
        while not fifo.isempty():
            s = fifo.remove()                      
            for q in [x for x in get_neighbours(img, s) if img[s] == img[x]]:
                if lab[q] == INIT:
                    lab[q] = curlab
                    fifo.add(q)
    curlab += 1   

print(lab)