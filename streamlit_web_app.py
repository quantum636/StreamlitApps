#import libararies
from ast import Param
from re import X
from argon2 import Parameters
from pyparsing import Word
from sklearn.multioutput import ClassifierChain
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt      
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Heading 
st.write("""
#Explore different ML model and datasets
we will check which one is best?
""")

# data name on sidebar
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

# Now put the name of classifier in an other box
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)

# Now we define a function to load a data set
def get_dataset(dataset_name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x,y        

# now we call the function and make equvilent to X,y variable
X, y = get_dataset(dataset_name)

# now we print the the shape of our dataset on app
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

# Next we add different classifier parameters into user parameters
def add_parameter_ui(classifier_name):
    params = dict # cerate an empty dictionary
    if classifier_name == 'SVM':
       C = st.sidebar.slider('C', 0.01, 10.0)
       params['C'] = C # its the degree of correct classification
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K # its the number of nearest neighbour
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth # depth of every tree the grow in the forest
        n_estimators = st.sidebar.slider('n_estimatores', 1, 100) 
        params['n_estimators'] = n_estimators # numbers of  trees
    return params         

    # Now we call the function make it equal to params variable
    params = add_parameter_ui(classifier_name)

# we make classifier baser on classifier_name and params
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], random_state=1234)
    return clf         

# NOW WE CALL THIS FUNCTION
clf = get_classifier(classifier_name, params)      

# now we split our data set in test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#we start our classifier training now
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#NOW WE check the accuracy of our model
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)


### Plot Datasets ###
# now we drew our features on a 2 dimentional plot 
pca = PCA(2)
X_projected = pca.fit_transform(X)


# now we slice our  data in  0 or 1 dimentional 
x1 = X_projected[:, 0]
x2 = X_projected[:, 1] 

fig = plt.figure()
plt.scatter(x1, x2,
         c=y, alpha=0.8,
         camp='viridis')

plt.xlabel('Principal component 1')
plt.ylabel('Principal componuent 2')
plt.colorbar() 

#plt.show()
st.pyplot(fig)
