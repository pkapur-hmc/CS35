#
#
# titanic.py
#
#

import numpy as np
from sklearn import datasets
from sklearn import cross_validation
import pandas as pd

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('titanic.csv', header=0)
df.head()
df.info()

# let's drop columns with too few values or that won't be meaningful
# Here's an example of dropping the 'body' column:
df = df.drop('body', axis=1)  # axis = 1 means column

# let's drop all of the rows with missing data:
df = df.dropna()

# let's see our dataframe again...
# I ended up with 1001 rows (anything over 500-600 seems reasonable)
df.head()
df.info()



# You'll need conversion to numeric datatypes for all input columns
#   Here's one example
#
def tr_mf(s):
    """ from string to number
    """
    d = { 'male':0, 'female':1, 'S':0, 'C':1, 'Q':2 }
    return d[s]

df['sex'] = df['sex'].map(tr_mf)  # apply the function to the column
df['embarked'] = df['embarked'].map(tr_mf)

# let's see our dataframe again...
df.head()
df.info()


# you will need others!


print("+++ end of pandas +++\n")

# import sys
# sys.exit(0)

print("+++ start of numpy/scikit-learn +++")

# We'll stick with numpy - here's the conversion to a numpy array

# extract the underlying data with the values attribute:
X_data = df.drop('survived',  axis=1).values  # everything except the 'survival' column
X_data = df.drop('ticket',  axis=1).values 
X_data = df.drop('cabin', axis=1).values 
X_data = df.drop('boat',  axis=1).values 
X_data = df.drop('home.dest',  axis=1).values 
y_data = df[ 'survived' ].values      # also addressable by column name(s)
y_data = df[ 'cabin' ].values    
y_data = df[ 'boat' ].values    
y_data = df[ 'home.dest' ].values    
#
# you can take away the top 42 passengers (with unknown survival/perish data) here:
#


X_test = X_data[0:42,0:14]              # the final testing data
X_train = X_data[43:,0:14]              # the training data

y_test = y_data[0:42]                  # the final testing outputs/labels (unknown)
y_train = y_data[42:]    



# feature engineering...
#X_data[:,0] *= 100   # maybe the first column is worth much more!
#X_data[:,3] *= 100   # maybe the fourth column is worth much more!


from sklearn.neighbors import KNeighborsClassifier

bestTrainScore = 0
bestTrainNeighbors = 0


#
# here, you'll implement the kNN model
#

L= [21,31,51,91, 175, 150, 131,211, 250, 265, 345, 366, 440, 480, 540, 120, 133, 780, 999, 1000, 987, 660, 888]

for i in L:
    knn = KNeighborsClassifier(n_neighbors=i)
    cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
        cross_validation.train_test_split(X_train, y_train, test_size= 0.1)
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
        cross_validation.train_test_split(X_train, y_train, test_size= 0.1)
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
  + how high were you able to get the average cross-validation (testing) score?



Then, include the predicted labels of the 12 digits with full data but no label:
Past those labels (just labels) here:
You'll have 12 lines:




And, include the predicted labels of the 10 digits that are "partially erased" and have no label:
Mention briefly how you handled this situation!?

Past those labels (just labels) here:
You'll have 10 lines:



"""