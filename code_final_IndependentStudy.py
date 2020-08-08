from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter


import pandas as pd

# Read the data from .csv file
data = pd.read_csv('commits.csv')
# COnvert into dataframe

df = pd.DataFrame(data)

# df1=df.loc[df['Repo'].isin(['csc216-001-P3-001','csc216-001-P3-002','csc216-001-P3-003','csc216-001-P3-004','csc216-001-P3-005','csc216-001-P3-006','csc216-001-P3-007','csc216-001-P3-008','csc216-001-P3-009','csc216-001-P3-010','csc216-001-P3-011','csc216-001-P3-012','csc216-001-P3-013'])]


# Chosse only Project3 as of now
df1 = df.iloc[19328:28211, :]
print(df1.shape)

# Remove initial commits made by instructor
df2 = df1[df1['commit_message'] != "b'Initial commit'"]

# Uncomment to know the shape of dataset
# print(df2.shape)
df2.sample(frac=1)

# create empty corpus, so that commit messages can be appended.
corpus = []
cor_words=[]
# get just commit messages into an array
for index, i in enumerate(df2.values):
    commit = i[1].split("'")
    line = commit[1].split("\\n")
    corpus.append(line[0])
    w=line[0].split(' ')
    for i in w:
        cor_words.append(i)


cor_freq=Counter(cor_words)
for k in list(cor_freq):
        if cor_freq[k] ==1:
            del cor_freq[k]


print(type(cor_freq))
filtered_feat=[]

for i in cor_freq.keys():
    filtered_feat.append(i)
"""
df3=df2.iloc[:,[0,1,2,3,4,5,6]]
df3.reset_index(inplace=True)
df3.drop(axis=0,labels=1)
df3.to_csv('Project.csv')
"""

import nltk
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# print(corpus)


# Make a vector of all words in the corpus
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)



# print(vectorizer.get_feature_names())


# Extract all words from the commit messages
words = vectorizer.get_feature_names()






# AFter vectorizing, remove Stop words
from nltk.corpus import stopwords

print("The length of bag or words before processing", len(words))
filtered_words = [word for word in words if word not in stopwords.words('english')]

# Uncomment to know the size of corpus after removing stop words.
# print("After processing ",len(filtered_words))


# Convert all the verbs into same tense (add/added will now be add)
from nltk.stem.wordnet import WordNetLemmatizer

changed_tense = []
for word in filtered_words:
    changed_tense.append(WordNetLemmatizer().lemmatize(word, 'v'))

# Extract unique words from the above step
changed_tense = np.array(changed_tense)
changed_tense = np.array(list(set(changed_tense)))

# Remove few words which are unnecessory
not_required = ['dd5e8759c80c36c96647ffe138c6ae6ecf205f90', 'man', '43', '', 'fnf', 'kj', 'fall2016', 'ball', 'csc216',
                '001', '80', 'edu', 'yellow', 'engr', 'yellowball', 'ncsu', 'github', 'black', 'nwere', 'nthe', 'nwhen',
                'nof','task list','to do list','ToDoList','black box test','observer','notifyObserver','addCategory',
                    'task','TaskList','ArrayList','LinkedList','file']

#proj_3_description=['task list','to do list','ToDoList','black box test','observer','notifyObserver','addCategory',
 #                   'task','TaskList','ArrayList','LinkedList','file']


new_array = [item for item in changed_tense if item not in not_required]

print(len(new_array))


print("The length of bag of words after processing", len(new_array))

# To count no of labels for each class, initiate all the possible labels with 0
B = 0
U = 0
N = 0
E = 0
T = 0

# Make a dataframe with columns as the words we made in above steps

# Uncomment to see the new words which are used are features in further steps
# print("Features")
# print(new_array)


# Create a new dataframe with above features as columns
df_p3 = pd.DataFrame(columns=new_array)

# read the file which has commit messages of Project3
new_csv = pd.read_csv('Project3.csv')
# Convert it into dataframe
df_new = pd.DataFrame(new_csv)

# Shuffle and take 300
df_new.sample(frac=1)
df_new = df_new.head(300)

print(df_new.shape)

# For every value in the dataframe just created, generate a binary matrix with the words from above filtered corpus.
for index, i in enumerate(df_new.values):

    a = i[2]
    b = a.split("'")
    c = b[1]
    # print(c,i[8])



    d = c.split(" ")
    #print(d)

    for j in d:
        t = WordNetLemmatizer().lemmatize(j, 'v')
        for k in filtered_feat:
            # try with new_array also
            # SO far new_array gave better results
            if (t == k):
                # print("yes")

                # Make the value of intersection of word from commit and corpus 1 if it's existing
                df_p3.at[index, k] = 1
    # Update the target column with the labels manually tagged.
    df_p3.at[index,'files_changed']=i[7]
    df_p3.at[index, 'target'] = i[8]

    if (i[8] == 'N'):
        N = N + 1
    if (i[8] == 'E'):
        E = E + 1
    if (i[8] == 'B'):
        B = B + 1
    if (i[8] == 'U'):
        U = U + 1
    if (i[8] == 'T'):
        T = T + 1

# for the intersections which are not 1 - i.e if the word from corpus isn't occurring in the commit messgae, make it 0.
df_p3.fillna(0, inplace=True)

## the dataframe of intersections - shape and distribution of target labels
print("The shape of Dataframe is ", df_p3.shape)
print("The counts for N, E, B, T, U are :", N, E, B, T, U)

## Pick the top 1500 (there are 1700 features - time being 1500 are used.)

# Split feature and target dataset.
X = df_p3.iloc[:, 0:1500]
X['File_changed']=df_p3['files_changed']
#X['Ass_specific']=df_p3['ass_specific']



y = pd.DataFrame(df_p3['target'])
# Uncomment to see the size and shape of featyres and target dataframes
print(X.shape, y.shape)



# make array of the target variable for label encoding
y_t = []

# Append values in this array  which are numbers rarther than letters.
for ind, i in enumerate(y.values):

    if (i == 'B'):
        y_t.append(1)
    elif (i == 'E'):
        y_t.append(2)
    elif (i == 'N'):
        y_t.append(3)
    elif (i == 'T'):
        y_t.append(4)
    elif (i == 'U'):
        y_t.append(5)
    else:
        y_t.append(5)

# Convert the features into matrix.
X_t = X.as_matrix()
# convert the array of target values (in numbers) to a dataframe
Y = pd.DataFrame(y_t)

print(Y.shape)

"""
# Uncomment to see the shape and size of the features matrix and target dataframe
# print(X_t.shape,y.shape,len(y_t),Y.shape)

# Split the features into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_t, Y, test_size=0.3, stratify=Y)

# X_train=pd.DataFrame(X_train)
# X_test=pd.DataFrame(X_test)

## Uncomment to see the shape of trianing and testing datasets
print(type(X_train), len(X_train), X_train.shape)
print(type(X_test), len(X_test), X_test.shape)
print(type(y_train), len(y_train), y_train.shape)
print(type(y_test), len(y_test), y_test.shape)

## Import thr packages for machine learning and metric-specific packages
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create a classifier for Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
test_l = np.array(y_test)

print("The test set after splitting using stratify")
print(test_l)
# print accuracy and other metrics
print("DECISION TREE")
print("Accuracy : ", accuracy_score(test_l, pred))
print("Classification report")
print(classification_report(test_l, pred))
print("Confusion Matrix")
print(confusion_matrix(test_l, pred))

# Import packages for Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

nb = GaussianNB()
nb.fit(X_train, y_train)
pred2 = nb.predict(X_test)

# print accuracy and other metrics
print("Naive Bayes")
print("Accuracy : ", accuracy_score(test_l, pred2))
print("Classification report")
print(classification_report(test_l, pred2))
print("Confusion Matrix")
print(confusion_matrix(test_l, pred2))

# Import packages for svm
from sklearn import svm

sv = svm.SVC(C=4, gamma=0.1)
sv.fit(X_train, y_train)
pred3 = sv.predict(X_test)
# Print accuracy and other metrics
print("SVM")
print("Accuracy : ", accuracy_score(test_l, pred3))
print("Classification Report")
print(classification_report(test_l, pred3))
print("Confusion Matrix")
print(confusion_matrix(test_l, pred3))
"""
# import packages for K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)

print(kmeans.labels_)
print(kmeans.cluster_centers_)

## Get the project 3 specic words (given by instructor)


print("COnfusion matrix fro Kmeans")
print(confusion_matrix(Y,kmeans.labels_))




