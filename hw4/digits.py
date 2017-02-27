#
#
# digits.py
#
#

import numpy as np
from sklearn import cross_validation
import pandas as pd

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('digits.csv', header=0)
df.head()
df.info()

# Convert feature columns as needed...
# You may to define a function, to help out:
def transform(s):
    """ from number to string
    """
    return 'digit ' + str(s)
    
df['label'] = df['64'].map(transform)  # apply the function to the column
print("+++ End of pandas +++\n")

# import sys
# sys.exit(0)

print("+++ Start of numpy/scikit-learn +++")

# We'll stick with numpy - here's the conversion to a numpy array
X_data = df.iloc[:,0:64].values        # iloc == "integer locations" of rows/cols
y_data = df[ 'label' ].values      # also addressable by column name(s)

#
# you can divide up your dataset as you see fit here...
#


#
# feature display - use %matplotlib to make this work smoothly
#
from matplotlib import pyplot as plt

def show_digit( Pixels ):
    """ input Pixels should be an np.array of 64 integers (from 0 to 15) 
        there's no return value, but this should show an image of that 
        digit in an 8x8 pixel square
    """
    print(Pixels.shape)
    Patch = Pixels.reshape((8,8))
    plt.figure(1, figsize=(4,4))
    plt.imshow(Patch, cmap=plt.cm.gray_r, interpolation='nearest')  # cm.gray_r   # cm.hot
    plt.show()
    
# try it!
# row = 53
# Pixels = X_data[row:row+1,:]
# show_digit(Pixels)
# print("That image has the label:", y_data[row])



X_test = X_data[0:22,0:64]              # the final testing data
X_train = X_data[22:,0:64]              # the training data

y_test = y_data[0:22]                  # the final testing outputs/labels (unknown)
y_train = y_data[22:]    







#
# feature engineering...
#

X_data[:,0] *= 0.01
X_data[:,7] *= 0.01
X_data[:,8] *= 0.01
X_data[:,31] *= 0.01
X_data[:,32] *= 0.1
X_data[:,39] *= 0.01
X_data[:,40] *= 0.1
X_data[:,47] *= 0.01
X_data[:,48] *= 0.01
X_data[:,55] *= 0.1
X_data[:,56] *= 0.01
X_data[:,8] *= 2
X_data[:,9] *= 2
X_data[:,10] *= 2
X_data[:,11] *= 2
X_data[:,63] *= 0.01

from sklearn.neighbors import KNeighborsClassifier

bestTrainScore = 0
bestTrainNeighbors = 0


#
# here, you'll implement the kNN model
#

L= [21,31,51,91, 175, 150, 131,211, 250, 265, 345, 366, 440, 480, 540, 120, 133, 780, 999, 1000, 1543, 1400, 987, 660, 888]

for i in L:
    knn = KNeighborsClassifier(n_neighbors=i)
    cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
        cross_validation.train_test_split(X_train, y_train, test_size=0.1)
    knn.fit(cv_data_train, cv_target_train) 
    trainScore = knn.score(cv_data_train, cv_target_train)
    if trainScore > bestTrainScore:
        bestTrainScore = trainScore
        bestTrainNeighbors = i

print("The best score is", bestTrainScore)
print("The best number of neighbors is", bestTrainNeighbors)

trainingAvg = 0
testingAvg = 0

#
# run cross-validation
#
for i in range(10):
    cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
        cross_validation.train_test_split(X_train, y_train, test_size=0.1)
    knn.fit(cv_data_train, cv_target_train) 
    trainingAvg += knn.score(cv_data_train,cv_target_train)
    testingAvg += knn.score(cv_data_test,cv_target_test)


#
# and then see how it does on the two sets of unknown-label data... (!)
#


knn.fit(X_train, y_train) 
print("\nCreated and trained a knn classifier")  #, knn

# here are some examples, printed out:
print("predicted outputs are")
print(knn.predict(X_test))

# and here are the actual labels (iris types)
print("and the actual labels are")
print(y_test)


"""
Comments and results:

Briefly mention how this went:
  + what value of k did you decide on for your kNN?

  we decided on number 21 of neighbors for our kNN 

  + how smoothly were you able to adapt from the iris dataset to here?

  The code was similar and we followed the same process, but had to think more 
  about the weights since there was a larger amount of data and there were columns
  with no data 

  + how high were you able to get the average cross-validation (testing) score?

  0.976

Then, include the predicted labels of the 22 digits with full data but no label:
Past those labels (just labels) here:
You'll have 12 lines:

['3' '3' '5' '5' '6' '5' '0' '2' '7' '3' '8' '4']


And, include the predicted labels of the 10 digits that are "partially erased" and have no label:
Mention briefly how you handled this situation!?

We handled the situation by comparing the predicted outcome with the show_digit 


You'll have 10 lines:

Past those labels (just labels) here:
['0' '0' '0' '1' '7' '3' '3' '4' '0' '7'] 


"""