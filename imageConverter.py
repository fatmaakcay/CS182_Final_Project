# Produces a CSV file of the test images in binary form
# Khalid Tawil

import image_pro.data as image
import config as cfg
import glob
import csv

emojis = ["hearts", "laugh", "sad", "smile"]


def convert_images():

    for i in range(4):

        # Open up the file to write in
        path = "./test_data/" + emojis[i] + "/converted.csv"
        f = open(path, 'w+')
        w = csv.writer(f)

        # Get the test cases
        test_cases = glob.glob('./test_data/' + emojis[i] + '/*.png')
        print 'Converting ' + emojis[i]

        for test_case in test_cases:
            # convert image
            data = image.binary_image(test_case, cfg.RES)
            data = image.convert_to_1d(data)

            # Write data into file
            w.writerow(data)

        # close the file
        f.close()
    print("files converted!")
