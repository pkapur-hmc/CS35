# ## Problem 2:  steganography
# 
# This question asks you to write two functions, likely with some helper functions, that will enable you
# to embed arbitrary text (string) messages into an image (if there is enough room!)

# For extra credit, the challenge is to be
# able to extract/embed an image into another image...

#
# You'll want to borrow from hw7pr1 for
#  + opening the file
#  + reading the pixels
#  + create some helper functions!
#  + also, check out the slides :-) 
#
# Happy steganographizing, everyone!
#
import numpy as np
from matplotlib import pyplot as plt
import cv2

# Part A: here is a signature for the decoding
# remember - you will want helper functions!

def lastbit(number):
    if number%2 == 0:
        return '0'
    else: 
        return '1'

def desteg_string( image ):
    """ takes in an image name with a hidden message
    function goes through pixels and extracts the lowest order bit from its channels
    combines these into a binary string, starting from the beginning, ending when 8 0's are found
    translates the binary string into the hidden message
    """
    raw_image = cv2.imread(image ,cv2.IMREAD_COLOR)
    #raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) #comment out this line when no need to convert
    new_image = raw_image.copy()
    rawbinary = ''
    binary = ''
    message = ''
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = new_image[row,col]
            rbit = lastbit(r)
            gbit = lastbit(g)
            bbit = lastbit(b)
            rawbinary += rbit
            rawbinary += gbit
            rawbinary += bbit
    print(rawbinary)
    i=0        
    
    while rawbinary[i:i+8] != '00000000':
        binary += rawbinary[i:i+8]
        chunk8 = rawbinary[i:i+8]
        print(chunk8)
        message += chr(int(chunk8,2))
        i+=8
    print('Last chunk is ' + str(rawbinary[i:i+8]))
    return message


def messagetobin(message):
    """takes in a string message, converts to ASCII binary form"""
    
    binary = ''
    for i in range(len(message)):
        character = ''
        character = bin(ord(message[i]))
        character = character[2:]
        numzeros = 8 - len(character)
        binary += '0'*numzeros + character
    binary += '0'* 16
    return binary


# Part B: here is a signature for the encoding/embedding
# remember - you will want helper functions!
def steganographize( image, message ):
    """ be sure to include a better docstring here! """
    raw_image = cv2.imread(image ,cv2.IMREAD_COLOR)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) #comment out this line when no need to convert
    new_image = raw_image.copy()
    newr = ''
    newg = ''
    newb = ''
    binmessage = messagetobin(message)    
    print(binmessage)
    num_rows, num_cols, num_chans = new_image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = new_image[row,col]
            #fix the r
            newr = str(r)
            newg = str(g)
            newb = str(b)
            
            if len(binmessage) != 0: 
                newr = newr[:-1] + binmessage[0]
                # for j in range(len(newr)-1):
                #     rbin = ''
                #     rbin += newr[j]    
                #     rbin += binmessage[0]
                binmessage = binmessage[1:]
                #newr = rbin
            #print(newr)
            #fix the g

            if len(binmessage) != 0: 
                newg= newg[:-1] + binmessage[0]
                binmessage = binmessage[1:]
                # for j in range(len(newg)-1):
                #     gbin = ''
                #     gbin += newg[j]
                #     gbin += binmessage[0]
                #     binmessage = binmessage[1:]
            #print(newg)
            #b
            
            if len(binmessage) != 0: 
                newb= newb[:-1] + binmessage[0]
                binmessage = binmessage[1:]
                # for j in range(len(newb)-1):
                #     bbin = ''
                #     bbin += newb[j]
                #     bbin += binmessage[0]
                #     binmessage = binmessage[1:]
                # newb = bbin
            #print("new b", newb)
            new_image[row,col] = [int(newr), int(newg), int(newb)]
    splitname = image.split('.')
    newname = splitname[0] + '_out.png'
    cv2.imwrite( newname, new_image)

    return
    
steganographize('spam.png', "hi")

m= desteg_string('spam_out.png')
print(m)
