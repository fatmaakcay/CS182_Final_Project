###############################################################################
# COMPSCI 182
# November 2015
# Khalid Tawil
#
# Takes in an image and reduces it to a 40x40 black and white image
# To see an image use im.show()
# USAGE: binary_image("colorfulsmiley.png")
###############################################################################
from PIL import Image

# Dimensions of our PNGs

def binary_image(file, size):
    # Loading file into memory
    im = Image.open(file)

    # Resizes the image
    im = im.resize(size, Image.ANTIALIAS)

    # Grayscales the image
    threshold = 250
    im = im.point(lambda p: p > threshold and 255)
    im = im.convert('1')
    im.show()

    # Converting our image into a 2-dimensional array of binary values
    data = list(im.getdata())
    binary_data =[]
    for i in range(size[0]):
        row = data[ (i * size[0]) : ((i+1) * size[0]) ]
        row = [1 if x==255 else x for x in row]
        binary_data.append(row)

    return binary_data

# Convert 2d array to 1d array
def convert_to_1d(bin_data):
    output = []
    for row in bin_data:
        for el in row:
            if el == 1:
                output.append(0)
            else:
                output.append(1)
    return output

def test():
    result = binary_image("../test_data/smile/010.png", (40, 40))

    # This just pretty prints
    s = [[str(e) for e in row] for row in result]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = ' '.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print '\n'.join(table)

    print "Output finished"

test()
