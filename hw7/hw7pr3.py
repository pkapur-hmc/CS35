# ## Problem 3:  green-screening!
# 
# This question asks you to write one function that takes in two images:
#  + orig_image  (the green-screened image)
#  + new_bg_image (the new background image)
#  
# It also takes in a 2-tuple (corner = (0,0)) to indicate where to place the upper-left
#   corner of orig_image relative to new_bg_image
#
# The challenge is to overlay the images -- but only the non-green pixels of
#   orig_image...
#

import numpy as np
import cv2

#
# Again, you'll want to borrow from hw7pr1 for
#  + opening the files
#  + reading the pixels
#  + create some helper functions
#    + defining whether a pixel is green is the key helper function to write!
#  + then, creating an output image (start with a copy of new_bg_image!)
#
# Happy green-screening, everyone! Include at least TWO examples of a background!
#

def checkGreen(rgbValues):
    """ Checks if the inputted rgb values encode something to be green, returns a boolean """
    r = rgbValues[0]
    g = rgbValues[1]
    b = rgbValues[2]
    if r < 60 and g < 90 and b < 60: 
        return True
	elif 0 < r < 160 and g > 70 and 0 < b < 150: 
        return True
    elif r > 90 and g > 90 and b > 90: 
        return False
	else: 
        return False 



# Here is a signature for the green-screening...
# remember - you will want helper functions!
def green_screen( orig_image, new_bg_image, corner=(0,0) ):
    """ Converts an image with a green screen background into an image with a new background
        specified by the input images to the function. Writes out the image into a png file called 
        greenScreened
    """
    o_raw = cv2.imread(orig_image,cv2.IMREAD_COLOR)
	o_image = cv2.cvtColor(o_raw, cv2.COLOR_BGR2RGB)
	n_raw = cv2.imread(new_bg_image,cv2.IMREAD_COLOR)
	n_image = cv2.cvtColor(n_raw, cv2.COLOR_BGR2RGB)

	rows, cols, chans = o_image.shape
    for r in range(rows):
		for c in range(cols):
			if not is_it_green(o_image[r,c]):
				n_image[r+corner[0],c+corner[1]]=o_image[r,c]
			else:
				pass

	n_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2RGB)
	cv2.imwrite('greenScreened.png', n_image)


