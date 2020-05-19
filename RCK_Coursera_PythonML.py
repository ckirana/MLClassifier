#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object 

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[6]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[7]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[8]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[10]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[11]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[12]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[13]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[14]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[15]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[16]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[17]:


X = Feature.astype('float64')
X[0:5]


# What are our lables?

# In[18]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[19]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#split dataset into training set and validation set
training_data, validation_data, training_label, validation_label = train_test_split(X, y, test_size = 0.2, random_state = 100)
print(len(training_data))
print(len(training_label))


# In[21]:


#finding the best k with highest accuracy, for k = 1 up to k = 20
accuracy_knn = []
for k in range(1,21):
  classifier = KNeighborsClassifier(n_neighbors = k) 
  classifier.fit(training_data, training_label)
  score = classifier.score(validation_data, validation_label)
  accuracy_knn.append(score)


# In[22]:


#plot k vs accuracy
k_list = range(1,21)

plt.plot(k_list, accuracy_knn)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Accuracy Evaluation for KNN")
plt.show()


# # Decision Tree

# In[23]:


from sklearn.tree import DecisionTreeClassifier

accuracy_dt = []
for k in range(1,21):
  tree = DecisionTreeClassifier(random_state = 1, max_depth = k)
  tree.fit(training_data, training_label)
  score = tree.score(validation_data, validation_label)
  accuracy_dt.append(score)

print(accuracy_dt)


# In[24]:


x = range(1, 21)

plt.plot(x, accuracy_dt)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Accuracy Evaluation for Decision Tree")
plt.show()


# # Support Vector Machine

# In[25]:


from sklearn.svm import SVC

best_svm = {'accuracy': 0, 'k': 0}
for k in range(1,21):
    clf = SVC(kernel = 'linear', C = k)
    clf.fit(training_data, training_label)
    score = clf.score(validation_data, validation_label)
    if score > best_svm['accuracy']:
        best_svm['accuracy'] = score
        best_svm['k'] = k

print(best_svm)


# # Logistic Regression

# In[26]:


from sklearn.linear_model import LogisticRegression

best_lr = {'accuracy': 0, 'k': 0}
for k in range(1,100):
    k = float(k)/100.0
    model = LogisticRegression(C = k, solver = 'liblinear')
    model.fit(training_data, training_label)
    score = model.score(validation_data, validation_label)
    if score > best_lr['accuracy']:
        best_lr['accuracy'] = score
        best_lr['k'] = k

print(best_lr)


# # Model Evaluation using Test set

# In[27]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[28]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[29]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[30]:


test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df.head()


# In[31]:


Feature_test = test_df[['Principal','terms','age','Gender','weekend']]
Feature_test = pd.concat([Feature_test,pd.get_dummies(test_df['education'])], axis=1)
Feature_test.drop(['Master or Above'], axis = 1,inplace=True)
Feature_test.head()


# In[32]:


X_test = Feature_test.astype('float64')
X_test[0:5]

y_test = test_df['loan_status'].values
y[0:5]

X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
X_test[0:5]


# In[33]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(training_data, training_label)
yhat_knn = knn.predict(X_test)

dt = DecisionTreeClassifier(random_state = 1, max_depth = 1)
dt.fit(training_data, training_label)
yhat_dt = dt.predict(X_test)

svm = SVC(kernel = 'linear', C = 1)
svm.fit(training_data, training_label)
yhat_svm = svm.predict(X_test)

lr = LogisticRegression(C = 0.01, solver = 'liblinear')
lr.fit(training_data, training_label)
yhat_lr = lr.predict(X_test)
yhat_lr_proba = lr.predict_proba(X_test)


# In[34]:


from sklearn.metrics import jaccard_similarity_score, f1_score, log_loss

labels = ['          KNN', 'Decision Tree', '          SVM', 'LogRegression']

for i, yhat in enumerate((yhat_knn, yhat_dt, yhat_svm)):
    label = labels[i]
    print(label, "Jaccard: %.2f F1-score: %.2f LogLoss: N/A" % (jaccard_similarity_score(y_test, yhat),
                                                    f1_score(y_test, yhat, average ='micro')))
print(labels[3], "Jaccard: %.2f F1-score: %.2f LogLoss: %.2f" % (jaccard_similarity_score(y_test, yhat_lr),
                                                                f1_score(y_test, yhat_lr, average = 'micro'),
                                                                log_loss(y_test, yhat_lr_proba)))


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
