# importing libraries
import numpy as np
import pandas as pd
from csv import reader 
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA





# Loading and cleaning the dataset.
opened_file = open('banknote_authentication.csv')
read_file = reader(opened_file)
banknotes = list(read_file)
bn = []
for row in banknotes:
    for item in row:
        str_item = str(item).rsplit(',') # breaks down the items in the row in separate strings
        bn.append(str_item)

col_names=['variance', 'skewness', 'kurtosis', 'entropy', 'class']

df = pd.DataFrame(bn, columns=col_names)
df = df.astype(float)
df = df.drop_duplicates()


# Splitting the dataset
X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modelling with tuned parameters 
# (Parameters are obtained thtough GridSearchCV. Please consult the notebook for detailed work)
knn_pipe = Pipeline(steps=[('standardscaler', StandardScaler()), ('pca', PCA()),
                ('kneighborsclassifier',
                 KNeighborsClassifier(leaf_size=20, n_neighbors=1))])

knn_pipe.fit(X_train, y_train)

# Saving the model
model = 'banknotes_auth_knn_classifier.pkl'
pickle.dump(knn_pipe, open(model, 'wb'))





