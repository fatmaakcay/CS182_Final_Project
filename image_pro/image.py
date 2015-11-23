###############################################################################
# COMPSCI 182
# November 2015
# Khalid Tawil
#
# Takes in an image and reduces it to a 200x200 black and white image
###############################################################################
from PIL import Image

# Dimensions of our PNGs
size = 200, 200

# Loading file into memory
im = Image.open("colorfulsmiley.png")

# NOTICE:
# Resizing first vs grayscaling first produces different results

# Resizes the image
im= im.resize((20,20), Image.BILINEAR)

# Grayscales the image
# The mode '1' attributes sets every pixel to either black or white
im = im.convert('1')

# Converting our image into a 2x2 array of binary values
# Still in beta...
bitmap = im.tobitmap()
print bitmap
