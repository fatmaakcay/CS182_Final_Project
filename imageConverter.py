# Produces a CSV file of the test images in binary form
# Khalid Tawil

import image_pro.data as image
import config as cfg
import glob
import csv

emojis = ["hearts", "laugh", "sad", "smile"]

# CSV Writer
f = open('binarytext.csv', 'w')
w = csv.writer(f)

for i in range(4):
    test_cases = glob.glob('./test_data/' + emojis[i] + '/*.png')
    print 'Loop ' + str(i)
    for test_case in test_cases:
        data = image.binary_image(test_case, cfg.RES)
        data = image.convert_to_1d(data)
        w.writerow(data)

f.close()
