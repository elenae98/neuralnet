#read a .bmp file to a numpy array

#need to download pillow for python3 
import numpy as np
from PIL import Image

im = Image.open("[8][100ms]X-Dir_Phase6-3.bmp")

data = np.array(im)

Image.Image.close(im)
