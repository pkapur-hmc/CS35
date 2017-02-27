#
# hw3pr2.py 
#
# Person or machine?  The rps-string challenge...
#
# This file should include your code for 
#   + extract_features( rps ),               returning a dictionary of features from an input rps string
#   + score_features( dict_of_features ),    returning a score (or scores) based on that dictionary
#   + read_data( filename="rps.csv" ),       returning the list of datarows in rps.csv
#
# Be sure to include a short description of your algorithm in the triple-quoted string below.
# Also, be sure to include your final scores for each string in the rps.csv file you include,
#   either by writing a new file out or by pasting your results into the existing file
#   And, include your assessment as to whether each string was human-created or machine-created
# 
#

"""
Short description of (1) the features you compute for each rps-string:
   Our scoring mechanism/what our function will do  
    We will get the length of the string, and we will determine the 
    amount of characters that are involved in strings with repeated sequences. 
    For instance, if "rpspsps" the string "pspsps" of six characters would be 
    considered for the numerator of the percentage value

    If the percentage is greater than .765, it is a human generated string
    If the percentage is less than .765, it is a computer generated string 

    (2) how you score those features and how those scores relate to "humanness" or "machineness":

    In class, we discovered that the strings that have a higher repetition within the strings 
    have a higher chance for being human given that computers randomly generate strings and it 
    is highly unlikely for a computer to be able to generate a string of repetitive 
    strings. We decided on 0.765 by trial and error. 

"""


# Here's how to machine-generate an rps string.
# You can create your own human-generated ones!

import random
import csv
import re

def gen_rps_string( num_characters ):
    """ return a uniformly random rps string with num_characters characters """
    result = ''
    for i in range( num_characters ):
        result += random.choice( 'rps' )
    return result

# Here are two example machine-generated strings:
rps_machine1 = gen_rps_string(200)
rps_machine2 = gen_rps_string(200)
# print those, if you like, to see what they are...




from collections import defaultdict

#
# extract_features( rps ):   extracts features from rps into a defaultdict
#
def extract_features( rps ):
    """ 
    Extract features is going to count the amount of r's s's and p's in all the 
    given strings. 
    """
    d = defaultdict( float )  # other features are reasonable
    number_of_s_es = rps.count('s')  # counts all of the 's's in rps
    number_of_p_es = rps.count('p') 
    number_of_r_es = rps.count('r')
    totalLength = len(rps)

    repetitions = []
    charRepCount = 0
    r = re.compile(r"(.+?)\1+")
    for match in r.finditer(rps):
        repetitions.append((match.group(1), len(match.group(0))/len(match.group(1))))
    
    countarep = 0
    for tup in repetitions:
        countarep += tup[1]
        charRepCount += (len(tup[0]))*tup[1]

    d['reps'] = float(charRepCount)/totalLength

    d['s'] = number_of_s_es/totalLength      # doesn't use them, however
    d['p'] = number_of_p_es/totalLength
    d['r'] = number_of_r_es/totalLength
    return d   # return our features... this is unlikely to be very useful, as-is

#
# score_features( dict_of_features ): returns a score based on those features
#
def score_features( dict_of_features ):
    """ 
    This function will take in the dictionary created in extract_features
    and it will return the score of the string
    """

    d = dict_of_features
    totalDifference = 0 
    countarep = 0

    for key,value in d.items(): 
        if key == 'reps':
            countarep = value
        else: 
            difference = abs(value - float(1.0/3.0))
            totalDifference+= difference 

    return countarep   # return a humanness or machineness score




#
# read_data( filename="rps.csv" ):   gets all of the data from "rps.csv"
#
def read_data( filename="rps.csv" ):
    """ 
    THis function will open the different strings in the csv file
    It will loop through the data and will only append to the list 
    the cell with the rps strings 
    """
    List_of_rows = [] 
    try:
        csvfile = open( filename , newline='' )  # open for reading
        csvrows = csv.reader( csvfile )              # creates a csvrows object

        for row in csvrows:                         # into our own Python data structure
            List_of_rows.append(row[3])

    except FileNotFoundError as e:
        print("File not found: ", e)
        return []
      # for now...
    return List_of_rows


def compute_scores():
    """
    Our scoring mechanism/what our function will do  
    We will get the length of the string, and we will determine the 
    amount of characters that are involved in strings with repeated sequences. 
    For instance, if "rpspsps" the string "pspsps" of six characters would be 
    considered for the numerator of the percentage value

    If the percentage is greater than .765, it is a human generated string
    If the percentage is less than .765, it is a computer generated string 
    """
    listOfStrings = read_data()
    listOfScores = []

    humanCount = 0
    compCount = 0
    countrp= 0

    for rpsString in listOfStrings:
        stringScore = []
        d = extract_features(rpsString)
        score = score_features(d)
        stringScore.append(score)
        if float(score) < .765:
            compCount += 1
            stringScore.append('c')
        else:
            humanCount += 1
            stringScore.append('h')

        listOfScores.append(stringScore)

    print(listOfScores)
    print (compCount)
    print(humanCount)

#
# you'll use these three functions to score each rps string and then
#    determine if it was human-generated or machine-generated 
#    (they're half and half with one mystery string)
#
# Be sure to include your scores and your human/machine decision in the rps.csv file!
#    And include the file in your hw3.zip archive (with the other rows that are already there)
#
