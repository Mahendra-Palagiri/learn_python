# 1️⃣ 📚 Training / Theory

# What is a Machine Learning Workflow?

# Think of it like cooking a recipe:
# 	1.	You decide what to cook → define the problem.
# 	2.	You get your ingredients → collect data.
# 	3.	You prepare the ingredients → clean & preprocess data.
# 	4.	You cook the dish → train the model.
# 	5.	You taste and adjust → evaluate the model.
# 	6.	You serve it → deploy the model.

# In ML, the main stages are:
# Machine Learning Workflow Reference
#
# | Stage            | Purpose                                                                              | Key Notes                              |
# |------------------|--------------------------------------------------------------------------------------|-----------------------------------------|
# | Define Problem   | Know exactly what you want the model to do (classification, regression, clustering)  | Avoid vague goals                       |
# | Collect Data     | Get relevant data from databases, APIs, CSV files, sensors, web scraping, etc.       | Data quality matters more than quantity |
# | Preprocess Data  | Handle missing values, encode categories, scale/normalize numbers                    | Garbage in → garbage out                |
# | Split Data       | Create training, validation, and test datasets                                       | Prevents overfitting                    |
# | Train Model      | Fit an algorithm to training data                                                    | Choice of algorithm depends on problem  |
# | Evaluate Model   | Measure performance with metrics (accuracy, precision, recall, RMSE, etc.)           | Helps you know if model is useful       |
# | Deploy & Monitor | Integrate into apps, websites, services; monitor for performance drift               | Not always needed in early learning     |


# Why Splitting Data Matters
# 	•	Train set: Model learns patterns here.
# 	•	Validation set: Tune parameters and choose models.
# 	•	Test set: Final unbiased performance check.

# If you skip splitting, your model will “cheat” by seeing all data before testing, giving overly optimistic results.

# Basic Python Tools for Workflow
# 	•	pandas → Data loading, cleaning, preprocessing
# 	•	scikit-learn → train_test_split, ML algorithms, evaluation metrics
# 	•	matplotlib/seaborn → Visualizing data and results

import sklearn.datasets as skd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# ~~~~~~~~~~~~~ 📥 Step 1 — Load the Dataset ~~~~~~~~~~~~~~``
	# •	Use sklearn.datasets.load_iris() to get the Iris data.
	# •	Turn the features into a DataFrame.
	# •	Turn the target into a Series.
	# •	Show the first 5 rows.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# data (sepal lenght, sepal width, petal length, petal width. --> in centimeters)
# target
# target_names (Setosa, Versicolour, Virgininica)
# feature_names )['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
# file name --> iris.csv
# data_module. --> sklearn.datasets.data
iris = skd.load_iris()

#fd = pd.DataFrame(iris['data'],columns=iris['feature_names'])
fd = pd.DataFrame(iris.data, columns=iris.feature_names) #Access the data using attributes / the above line is equally good
print(fd)

ts  = pd.Series(iris['target'], name='target')

print("\ntarget data \n",iris['target'])
print("\n Series Data of target\n",ts)

print("\n First 5 rows of data\n", fd.head(5))
print("\n Check same number of samples\n",ts.value_counts()) #Check if data is balanced equally

fdc = fd.copy()
#fdc["Species"] = ts.map({0:'setosa',1:'versicolor',2:'virginica'})
fdc['Species'] = iris.target_names[ts] #Efficient way of mapping (instead of provding the values manually like above)

sns.pairplot(fdc,hue='Species',diag_kind='kde')
plt.show()

# ~~~~~~~~~~~~~ 📥 Step 2 — Inspect the dataset ~~~~~~~~~~~~~~``
	# •	Count how many samples there are for each flower type.
	# •	Check the number of rows and columns.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print(f"\nAssuming a uniform data set samples for  flower types  --> '{iris.target_names}' --> , {ts.value_counts()}")
print(f"\n Number of rows and colums \n {len(fd)},  {len(fd.columns)}")
#or other way to get data
rows, cols = fd.shape
print(f"\n Numer of rows --> {rows}, Number of Cols --> {cols}")
print(f"\n Numer of rows --> {fd.shape[0]}, Number of Cols --> {fd.shape[1]}")


# ~~~~~~~~~~~~~ 📥 Step 3 — Split the dataset ~~~~~~~~~~~~~~``
# 	🔎 What does it mean?
# 	•	You take your full dataset and divide it into two (or sometimes three) parts:
# 	1.	Training set → The model learns patterns here.
# 	2.	Test set → Used at the very end to check how well the model generalizes.
# 	3.	(Optional) Validation set → Used while tuning parameters (sometimes replaced by cross-validation).

# ⸻

# ⚖ Why do we split?
# 	•	If you train and test on the same data, the model may just memorize answers (overfitting).
# 	•	By splitting, we simulate unseen data, so we know if the model can generalize.

# ⸻

# 📐 Typical Ratios
# 	•	80/20 split → 80% training, 20% testing (common starting point).
# 	•	70/15/15 split → Training, validation, testing.
# 	•	Larger datasets → You can afford to keep more for testing.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


X_train,X_test,Y_train,Y_test = train_test_split(fd,ts,test_size=0.2,random_state=42,stratify=ts)
# X :: Features  (input variables)  --> features of flower petal and sepal length and width --> Stored in 'fd'
# Y :: Target (output variables) --> flower types data  --> stored in 'ts' 
# test size of 0.2 indicates 80% of data to use for training and 20% of data for testing
# random_state=42  (ensures the data is picked up consistently in same fashion to ensure we get unfirom results)
# stratify=ts  (test data is equally distributed for all data elements (in this example ts which has flower type data))

# iris data set is considered the hello world of ML  
# It has 4 data sets
# 		* Features data (iris.data) --> Features
# 		* meta data for feature data (iris.feature_names) --> Feature Names
# 		* Target data (iris.target) --> Flower samples (the actual target). --> iris.data[0] i.e. first row is assoicated with iris.target[0] i.e. first row of Flower
# 		* meta data for target data (iris.target_names) --> target Names. (Index of Each flower in iris.target_names is tracked in iris.target i.e. iris.target_names[0] i.e. --> 'Setosa' has index of '0' and all relevant data is tracked with index '0' in iris.target)

print(X_train.shape)
print(X_test.shape)
print(Y_train.value_counts())
print(Y_test.value_counts())
print(len(X_train)/len(fdc)*100)


# ~~~~~~~~~~~~~ 📥 Step 4 — Choose a simple model ~~~~~~~~~~~~~~``
	# Evaluate Models
    # As iris data set fits the classification model, we are evaluting the following 3 model. (How do we know that iris is a classification --> data set is about the flowers and our model should predict what type of flower based on data)
    # KNN (K-Nearest Neighbour) --> Ask your neighbour
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.neighbors import KNeighborsClassifier   #Import the model
from sklearn.linear_model import LogisticRegression
from sklearn. tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report, ConfusionMatrixDisplay


model = KNeighborsClassifier(n_neighbors=5) #Create the Model (We are asking 5 neighbours in the model) when the model classifies a flower, it looks at the 5 closest data points (neighbors) in the training set and uses majority vote to decide the class
model.fit(X_train,Y_train) #Run the model (now the model has trained on the data)
Y_pred = model.predict(X_test) #Predict on the test data 

Accuracy  = accuracy_score(Y_test, Y_pred) #Check the accuracy of prediction. i.e Compare true lables (Y_test) with predicted labels (Y_pred)
print(Accuracy)
Report = classification_report(Y_test,Y_pred,target_names=iris.target_names) #Generate a report
print(Report)

# Generates a detailed performance report:
# 	•	precision (how often predicted positives were correct),
# 	•	recall (how many actual positives were found),
# 	•	f1-score (balance of precision & recall),
# 	•	support (number of samples per class).

#  ------- Output ---------
#               precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        10
#   versicolor       1.00      1.00      1.00        10
#    virginica       1.00      1.00      1.00        10

#     accuracy                           1.00        30
#    macro avg       1.00      1.00      1.00        30
# weighted avg       1.00      1.00      1.00        30

ConfusionMatrixDisplay.from_estimator(model,X_test,Y_test)
plt.show()



