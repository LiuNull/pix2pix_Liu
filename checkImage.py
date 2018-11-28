import numpy as np
import os

from PIL import Image

a = Image.open("/home/hsx/project/pix2pixHD_NoFeat/pix2pixHD/frankfurt_000000_000294_gtFine_labelTrainIds.png")

np.savetxt("image2.txt",a)
