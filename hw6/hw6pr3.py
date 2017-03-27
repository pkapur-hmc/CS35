# coding: utf-8


#
# hw6 problem 3
#

## Problem 3: Paraphrasing!

import textblob
from nltk.stem import WordNetLemmatizer
import nltk
lemmatizer = WordNetLemmatizer()


# A starter function that substitutes each word with it's top match
#   in word2vec.  Your task: improve this with POS tagging, lemmatizing, 
#   and/or at least three other ideas of your own (more details below)
#
def paraphrase_sentence( sentence, model ):
    """ This function returns a *very bad* paraphrased version of a sentence by:
        1. Checking to make sure the root of the word being replced is not the same as the word replacing it
            (using lemmatization)
        2. The replacement word cannot have the same first letter as the original
        3. There are no replacement words with the letter e
    """
    blob = textblob.TextBlob( sentence )
    print("The sentence's words are")
    LoW = blob.words
    print(LoW)

    NewLoW = []

    for w in LoW:
        if w not in model:
            NewLoW += [w]
        else:
            w_alternatives = model.most_similar(positive=[w], topn=100)
            for i in range(100):
                check = str(w_alternatives[i][0].lower())
                if lemmatizer.lemmatize(check,'v') != lemmatizer.lemmatize(str(w),'v') and \
                    w.lower()[0] != check[0] and \
                    'e' not in check:
                    alternative = w_alternatives[i][0]
                    break
            NewLoW += [alternative]
    
    print( "NewLoW is" )
    returnString = " ".join(NewLoW)
    return returnString

"""

(3)

Examples:

1.(bad) Input: 'Why are you so bad at paraphrasing'
Output: 'How now I too good in quoting'

2. Input: 'i like nice'
Output: 'u crazy good'

3.(good) Input: 'human is nice'
Output: 'mankind now good'

4. Input: 'an apple a day keeps the doctor away'
Output: 'this fruit a morning stays in physician out'

"""


# 
# Once the above function is more sophisticated (it certainly does _not_ need to be
#   perfect -- that would be impossible...), then write a file-paraphrasing function:
#
"""
(4)
"""
def paraphrase_file(filename, model):
    """ opens a plain-text file, reads its contents, tokenizes it into sentences, paraphrases all of 
    the sentences, and writes out a new file containing the full, paraphrased contents with the 
    word paraphrased in its name
    """
    with open(filename, 'r') as myfile:
        data=myfile.read().replace('\n', '')

    LoS = nltk.sent_tokenize(data.lower())
    output = []
    for sentence in LoS:
        paraphrased = paraphrase_sentence(sentence, model)
        output.append(paraphrased)

    returnString = "\n".join(output)
    outputFile = open('test_paraphrased.txt', 'w')
    outputFile.write(returnString)
    outputFile.close()



#
# Results and commentary...
#

# The exercise was confusing to navigate at first, but once we understoodhow the different 
# parts of TextBlob worked, it was easier to code the different parts. At first we would get 
#words that were extremely similar, but then it was hard to keep the words to make sense 
# in the grammatical context. We explain what we did in the docstrings 



# (1) Try paraphrase_sentence as it stands (it's quite bad...)  E.g.,
#         Try:    paraphrase_sentence("Don't stop thinking about tomorrow!", m)
#         Result: ['Did', "n't", 'stopped', 'Thinking', 'just', 'tonight']

#     First, change this so that it returns (not prints) a string (the paraphrased sentence),
#         rather than the starter code it currently has (it prints a list) Thus, after the change:

#         Try:    paraphrase_sentence("Don't stop thinking about tomorrow!", m)
#         Result: "Did n't stopped Thinking just tonight"  (as a return value)

#     But paraphrase_sentence is bad, in part, because words are close to variants of themselves, e.g.,
#         + stop is close to stopped
#         + thinking is close to thinking





# (2) Your task is to add at least three things that improve this performance (though it
#     will necessarily still be far from perfect!) Choose at least one of these two ideas to implement:

#     #1:  Use lemmatize to check if two words have the same stem/root - and _don't_ use that one!
#             + Instead, go _further_ into the similarity list (past the initial entry!)
#     #2:  Use part-of-speech tagging to ensure that two words can be the same part of speech

#     Then, choose two more ideas that use NLTK, TextBlob, or Python strings -- either to guard against
#     bad substitutions OR to create specific substitutions you'd like, e.g., just some ideas:
#        + the replacement word can't have the same first letter as the original
#        + the replacement word is as long as possible (up to some score cutoff)
#        + the replacement word is as _short_ as possible (again, up to some score cutoff...)
#        + replace things with their antonyms some or all of the time
#        + use the spelling correction or translation capabilities of TextBlob in some cool way
#        + use as many words as possible with the letter 'z' in them!
#        + don't use the letter 'e' at all...
#     Or any others you might like!





# (3) Share at least 4 examples of input/output sentence pairs that your paraphraser creates
#        + include at least one "very successful" one and at least one "very unsuccessful" ones






# (4) Create a function paraphrase_file that opens a plain-text file, reads its contents,
#     tokenizes it into sentences, paraphrases all of the sentences, and writes out a new file
#     containing the full, paraphrased contents with the word paraphrased in its name, e.g.,
#        + paraphrase_file( "test.txt", model )
#             should write out a file names "test_paraphrased.txt"  with paraphrased contents...
#        + include an example file, both its input and output -- and make a comment on what you
#             chose and how it did! 











# (Optional EC) For extra-credit (up to +5 pts or more)
#        + [+2] write a function that takes in a sentence, converts it (by calling the function above) and
#          then compares the sentiment score (the polarity and/or subjectivity) before and after
#          the paraphrasing
#        + [+3 or more beyond this] create another function that tries to create the most-positive or
#          most-negative or most-subjective or least-subjective -- be sure to describe what your
#          function does and share a couple of examples of its input/output...






