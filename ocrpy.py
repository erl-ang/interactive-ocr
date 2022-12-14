description = "Script to augment League of Nations images"
import os
import numpy as np

import argparse
from argparse import RawTextHelpFormatter
import cv2
from PIL import Image

#img_path='tmp/GPO-CRECB-1920-pt3-v59-11-1-004.png'

def cli():
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--file",
        required=True,
        type=str,
        help="Name of file")

    args = parser.parse_args()
    return (args.file)


def main(img_path):
    # Read image using opencv
    im = Image.open(img_path)
    length_x, width_y = im.size
    factor = max(1, int(1800 / length_x))
    size = factor * length_x, factor * width_y
    # size = (1800, 1800)
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(img_path, dpi=(300, 300))
    img = cv2.imread(img_path)

    # Extract the file name without the file extension
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]

    # Create a directory for outputs
    #output_path = os.path.join('x', file_name)
    output_path = 'x'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Rescale the image, if needed.
    #img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(img, kernel, iterations=1)
    #img = cv2.erode(img, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    #img = cv2.GaussianBlur(img, (5, 5), 0)
    #img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Save the filtered image in the output directory
    save_path = os.path.join(output_path, file_name + ".png")
    cv2.imwrite(save_path, img)

if __name__ == '__main__':
    img_path  = cli()
    main(img_path)