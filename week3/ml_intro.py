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
import pandas as pd
import numpy as np

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

ts  = pd.Series(iris['target'], name='target')

print("\ntarget data \n",iris['target'])
print("\n Series Data of target\n",ts)

print("\n First 5 rows of data\n", fd.head(5))



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
