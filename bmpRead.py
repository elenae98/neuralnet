#a way to read a specific .bmp file to a bytes object

#need to download pillow for python3 
from PIL import Image
im = Image.open("[8][100ms]X-Dir_Phase6-3.bmp")

im_bytes = im.tobytes()

Image.Image.close(im)