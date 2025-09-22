## 🐍 **Mahi's** Journey into Python and AI Land  
A motivation to learn AI using Python

<details>
<summary>📜 **Program Overview** — *(Click to Expand)*</summary>
<br/>

| Month | Week | Topic | Math Skills Introduced | Status | Completion Month |
|-------|------|-------|------------------------|--------|------------------|
| **1 – Python Foundations**  | Week 1 | Python basics | None (focus on syntax & logic) | ✅ Done| July 2025 |
|       | Week 2 | NumPy, Pandas, Seaborn | Mean, median, variance, standard deviation | ✅ Done | Aug 2025 |
| **2 – Data Analysis & Visualization**| Week 3 | Intro to ML workflow | Basic probability, correlation & covariance | 🔄 In Progress |  |
|       | Week 4 | Advanced Pandas & Viz | Grouped statistics, weighted averages | 📅 To Do |  |
| **3 – ML Foundations**  | Week 5 | Classification deep dive | Logistic function, odds, log-odds | 📅 To Do |  |
|       | Week 6 | Regression deep dive | Linear equations, least squares, RMSE formula | 📅 To Do |  |
|       | Week 7 | Model selection & validation | Bias-variance tradeoff, cross-validation math | 📅 To Do |  |
|       | Week 8 | Mini capstone | Consolidation of above | 📅 To Do |  |
| **4 – Deep Learning Foundations**  | Week 9 | Neural networks basics | Derivatives, chain rule, gradient descent math | 📅 To Do |  |
|       | Week 10 | PyTorch basics | Matrix multiplication, dot products | 📅 To Do |  |
|       | Week 11 | CNNs | Convolution operation math | 📅 To Do |  |
|       | Week 12 | Mini capstone | Consolidation of above | 📅 To Do |  |
| **5 – Applied AI**  | Week 13 | NLP | Probability distributions, cosine similarity | 📅 To Do |  |
|       | Week 14 | Computer vision | Image filter kernels, normalization | 📅 To Do |  |
|       | Week 15 | Time series | Seasonal decomposition, autocorrelation | 📅 To Do |  |
|       | Week 16 | Mini capstone | Consolidation of above | 📅 To Do |  |
| **6 – Deployment & Final Capstone**| Week 17 | Deployment basics | None new | 📅 To Do |  |
|       | Week 18 | MLOps | None new | 📅 To Do |  |
|       | Week 19–20 | Final capstone | Apply all math learned | 📅 To Do |  |

</details>
</br>

<details>
<summary>📅 Week-by-Week Learning Journey (Click to Expand...)</summary>
<br/>

<details>
<summary>📘 Week 1: Python Basics</summary>
<br/>

- **Topics Covered**: Python syntax, variables, functions, I/O  
- **Status**: ✅ Completed  
- **Days Spent**: 3  
- **Time Frame**: July 2025
- **Basic softwares setup**:
    > ### Install Homebrew (Package Manager for Mac)
        ```bash
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        ### To verify 
        brew --version


    > ### Install Python  
        ```bash
        brew install python
            
        ### To verify 
        python3 --version
        pip3 --version

    > ### Install VS Code  
        ```bash
        brew install --cask visual-studio-code
    
    > ### Install Jupyter Lab  
        ```bash
        pip3 install jupyterlab

- **Python Virtual Environment and packages**:

        ```bash
        python3 -m venv .venv

        ### activating virutal environment
        source .venv/bin/activate

        ### Installing packages
        pip install numpy pandas matplotlib jupyterlab
        pip install scikit-learn seaborn

        pip freeze > requirements.txt ##records all the packages in requirements.txt file
        pip install -r requirements.txt ##to install all pacakages present in requirements.txt file

        ### when working with fodlers in python
        ** Any folder in python can be turned into module by adding __init__.py
        touch __init__.py  ##command to create the file

- **Type of lists**:
    > ### List
        Ordered: ✅   | Changeable : ✅  | Use Case : Sequences, item collections

    > ### Tuple
        Ordered: ✅   | Changeable : ❌  | Use Case : Fixed structure, return multiple
    
    > ### Dictionary
        Ordered: ❌ (3.6+ preserves)   | Changeable : ✅  | Use Case : ast lookup by key, named data

  ```python
    🔹 1. Tuple → For fixed pair of items (position matters, no labels)

    ✅ Use When:
        •	The two elements are tightly linked
        •	The meaning is implied by position
        •	You don’t plan to change them

    🧠 Example:
    coordinates = (40.7128, -74.0060)  # (latitude, longitude)
    name_pair = ("Alice", "Bob")       # two people


    🔹 2. List → For a sequence of items (even if just two), possibly growing

    ✅ Use When:
        •	You might add/remove more items later
        •	Order matters
        •	You don’t need labels

    🧠 Example:
    scores = [88, 91]

    🔹 3. Dictionary → When each element has a meaning (a label)

    ✅ Use When:
	•	You want to name each value
	•	The meaning is not obvious from order
	•	You want to access by key, not index

    🧠 Example:
    student = {"name": "Mahendra", "score": 95}


### Week 1 — Day-by-Day Breakdown

## Week 1 — Day 1 ✅

- Focus: Introduction to Python syntax, variables, and basic I/O  
- Learned the basics of Python syntax and how to declare variables.  
- Practiced printing output and reading user input.  
- Understood data types: strings, integers, floats, and booleans.

✅ Outcome: Comfortable with writing simple Python scripts and handling basic input/output.

## Week 1 — Day 2 ✅

- Focus: Functions and control flow  
- Learned how to define and call functions.  
- Explored conditional statements (`if`, `elif`, `else`).  
- Practiced loops: `for` and `while`.  
- Understood function parameters and return values.

✅ Outcome: Able to write reusable code blocks and control program flow effectively.

## Week 1 — Day 3 ✅

- Focus: Data structures - lists, tuples, dictionaries, and introduction to classes  
- Explored Python collections: lists (mutable), tuples (immutable), dictionaries (key-value pairs).  
- Learned when to use each data structure with examples.  
- Introduced basic class syntax and object-oriented concepts.  
- Created simple classes with attributes and methods.

✅ Outcome: Gained foundational understanding of Python data structures and basic object-oriented programming.

## 📑 Week 1 — Question Bank & Coding Challenges

### 🟢 Easy (15 Qs + 5 Mini Challenges)

#### Questions
1.	What are Python’s basic data types?
2.	How do you declare variables in Python?
3.	How is input() used and how do you convert input to an integer?
4.	Difference between = and ==.
5.	What is the purpose of comments in Python?
6.	Explain the difference between a list and a tuple.
7.	What is a dictionary in Python?
8.	How do you access a value in a dictionary safely?
9.	Difference between print() and return.
10.	Can a function return multiple values? If yes, how?
11.	What is __init__ in a class?
12.	How do you call a method from a class instance?
13.	Explain the scope of local vs global variables.
14.	What is indentation and why is it important in Python?
15.	What are mutable vs immutable objects?

#### Mini Coding Challenges
1.	Write a program that prints “Hello, World!”.
2.	Ask the user for their name and age, and print a message back.
3.	Write a program to find the largest of two numbers.
4.	Swap two numbers without using a third variable.
5.	Write a program that takes a number and prints whether it is even or odd.

⸻------------------------------------------------------------------------------------

### 🟡 Medium (15 Qs + 5 Mini Challenges)

#### Questions
1.	Write a function that checks if a number is prime.
2.	How do default arguments in functions work?
3.	What’s the difference between positional and keyword arguments?
4.	Explain *args and **kwargs.
5.	How do you define a class in Python?
6.	What’s the difference between a class attribute and an instance attribute?
7.	What is method overriding in Python?
8.	How does Python handle boolean values in conditionals?
9.	What happens if you call a function without a return statement?
10.	How can you check the type of a variable?
11.	Write a function that reverses a list.
12.	How do you remove an element from a list by value and by index?
13.	What is the difference between is and ==?
14.	How does Python handle NoneType?
15.	What is string concatenation and how is it done?

#### Mini Coding Challenges
1.	Write a function to calculate factorial of a number.
2.	Write a function that returns the sum of all elements in a list.
3.	Write a program that prints the multiplication table of a number.
4.	Write a function that checks if a word is a palindrome.
5.	Implement a simple calculator with functions for add, subtract, multiply, divide.

⸻------------------------------------------------------------------------------------

### 🔴 Tricky (15 Qs + 5 Mini Challenges)

#### Questions
1.	How does Python handle memory management?
2.	What are Python namespaces?
3.	Explain the concept of scope with LEGB (Local, Enclosing, Global, Built-in).
4.	What is recursion? Give an example.
5.	Explain shallow copy vs deep copy with examples.
6.	What are Python modules and how do you import them?
7.	How do you create a package in Python?
8.	What happens if two functions have the same name?
9.	What are Python’s special methods (dunder methods)?
10.	Explain how __str__ and __repr__ differ.
11.	How do you define private vs public attributes in a class?
12.	What is operator overloading in Python?
13.	Explain Python’s dynamic typing.
14.	How does Python’s garbage collector handle circular references?
15.	What’s the difference between id() and hash()?

#### Mini Coding Challenges
1.	Write a recursive function to compute Fibonacci numbers.
2.	Implement a class Rectangle with methods to compute area and perimeter.
3.	Write a program that simulates a simple bank account (deposit, withdraw, balance).
4.	Implement a function that removes duplicates from a list without using set().
5.	Write a custom iterator class that yields squares of numbers up to N.


</details>

<details>
<summary>📗 Week 2: NumPy, Pandas, Data Viz</summary>
<br/>

- **Topics Covered**: NumPy arrays, Pandas DataFrames, Matplotlib, Seaborn  
- **Status**: ✅ Completed  
- **Days Spent**: 4  
- **Time Frame**: Aug 2025

- **Core libraries and setup**:
    > ### Install required libraries
        ```bash
        source .venv/bin/activate   # activate virtual environment
        pip install numpy pandas matplotlib seaborn jupyterlab
        pip freeze > requirements.txt
        ```

- **Key Skills**:
    > ### NumPy  
      - Array creation, slicing, broadcasting  
      - Statistical functions: mean, std, variance  
      - Random: seed, rand, randint  
      - Split: split vs array_split  
      - Transpose for 2D and ND arrays

    > ### Pandas  
      - Series & DataFrame creation  
      - Column selection, filtering, sorting (multi-column)  
      - GroupBy and aggregation  
      - Merge and joins  
      - Date parsing with `parse_dates`  
      - Boolean masks and conditional columns with `np.where` / `np.select`

    > ### Plotting  
      - Matplotlib: bar charts, line charts, histograms  
      - Seaborn: barplot, histplot (for reference only)  
      - Sorting data before plotting for chronological trends

- **Examples**:

    > **NumPy Example**
    ```python
    import numpy as np

    arr = np.array([10, 20, 30, 40, 50])
    filtered = arr[arr > 25]
    print("Filtered:", filtered)
    print("Mean:", np.mean(arr))
    ```

    > **Pandas Example**
    ```python
    import pandas as pd

    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Score': [85, 92, 78],
        'Grade': ['A', 'A', 'B']
    })

    passed = df[df['Score'] >= 80]
    print("Passed:\n", passed)
    print("Average Score by Grade:\n", df.groupby('Grade')['Score'].mean())
    ```

    > **Plotting Example (Matplotlib)**
    ```python
    import matplotlib.pyplot as plt

    scores = {'Alice': 85, 'Bob': 92, 'Charlie': 78}
    plt.bar(scores.keys(), scores.values(), color='skyblue')
    plt.xlabel("Name")
    plt.ylabel("Score")
    plt.title("Scores by Student")
    plt.show()
    ```

    > **Plotting Example (Seaborn — Reference)**
    ```python
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Score': [85, 92, 78, 88]
    })

    sns.barplot(x='Name', y='Score', data=df, palette='pastel')
    plt.title("Scores by Student")
    plt.show()
    ```

### Week 2 — Day-by-Day Breakdown

## Week 2 — Day 1 ✅

- Focus: NumPy basics and array operations  
- Learned how to create NumPy arrays and perform slicing and indexing.  
- Explored broadcasting rules and array arithmetic.  
- Practiced statistical functions like mean, standard deviation, and variance.

✅ Outcome: Able to manipulate numerical data efficiently using NumPy arrays.

## Week 2 — Day 2 ✅

- Focus: Advanced NumPy and random operations  
- Explored random number generation with seed, rand, and randint.  
- Learned difference between `split` and `array_split`.  
- Practiced transposing arrays and working with multi-dimensional data.

✅ Outcome: Gained deeper control over NumPy array manipulation and randomness.

## Week 2 — Day 3 ✅

- Focus: Pandas DataFrames and Series  
- Learned to create Series and DataFrames from dictionaries and lists.  
- Explored column selection, filtering rows, sorting by multiple columns.  
- Practiced grouping data and applying aggregation functions.

✅ Outcome: Confident in handling tabular data with Pandas.

## Week 2 — Day 4 ✅

- Focus: Data visualization with Matplotlib and Seaborn  
- Created bar charts, line charts, and histograms using Matplotlib.  
- Used Seaborn for enhanced visualizations and styling.  
- Learned importance of sorting data before plotting for clarity.

✅ Outcome: Able to create clear and informative data visualizations using popular Python libraries.

## 📑 Week 2 — Question Bank & Coding Challenges

### 🟢 Easy (15 Qs + 5 Mini Challenges)

#### Questions
1.	What is a NumPy array and how is it different from a Python list?
2.	How do you create a NumPy array?
3.	What is broadcasting in NumPy?
4.	How do you select rows and columns in a Pandas DataFrame?
5.	What is the difference between a Series and a DataFrame in Pandas?
6.	How do you create a Pandas DataFrame from a dictionary?
7.	How do you plot a simple bar chart using Matplotlib?
8.	What is the purpose of plt.show() in Matplotlib?
9.	How do you install and import Seaborn?
10.	How do you filter a DataFrame based on a condition?
11.	What is the use of np.mean() and np.std()?
12.	How do you transpose a NumPy 2D array?
13.	What’s the difference between .loc[] and .iloc[] in Pandas (basic)?
14.	What is a histogram used for in data visualization?
15.	How do you save a Matplotlib plot as an image file?

#### Mini Coding Challenges
1.	Create a NumPy array of numbers from 1 to 10 and print it.
2.	Create a Pandas DataFrame with columns “Name” and “Age” and print it.
3.	Plot a bar chart of the number of students in different classes using Matplotlib.
4.	Filter a DataFrame to show only rows where the score is above 80.
5.	Use Seaborn to create a histogram of a given dataset.

⸻----------------------------------------------------------------------------------------------

### 🟡 Medium (15 Qs + 5 Mini Challenges)

#### Questions
1.	Explain the difference between np.split and np.array_split.
2.	How do you use groupby in Pandas and what is it used for?
3.	What is a Boolean mask in Pandas and how is it applied?
4.	How do you merge two DataFrames in Pandas?
5.	What are some common aggregation functions used with groupby?
6.	How do you handle missing data in Pandas?
7.	Explain the difference between loc and iloc in Pandas.
8.	How do you customize the color palette in Seaborn plots?
9.	How do you save a Matplotlib plot to a file?
10.	What is the difference between plt.plot() and sns.lineplot()?
11.	How do you apply multiple conditions to filter a DataFrame?
12.	What is vectorization in NumPy and why is it faster?
13.	How do you reset the index of a Pandas DataFrame?
14.	Explain how .pivot() differs from .melt() in Pandas.
15.	How do you add a calculated column in Pandas using np.where?

#### Mini Coding Challenges
1.	Write a function to compute the mean and standard deviation of a NumPy array.
2.	Group a DataFrame by a categorical column and compute the average of a numerical column.
3.	Merge two DataFrames on a common key and display the result.
4.	Create a scatter plot using Seaborn with custom colors and labels.
5.	Write a program to fill missing values in a DataFrame column with the column mean.

⸻----------------------------------------------------------------------------------------------

### 🔴 Tricky (15 Qs + 5 Mini Challenges)

##### Questions
1.	How does NumPy handle memory differently than Python lists?
2.	What is the difference between views and copies in NumPy arrays?
3.	Explain the concept of chaining assignments in Pandas and why it is discouraged.
4.	How do you optimize Pandas operations for large datasets?
5.	What are categorical data types in Pandas and why use them?
6.	How does Seaborn integrate with Matplotlib under the hood?
7.	Explain how pivot tables work in Pandas.
8.	How do you create multi-index DataFrames and why are they useful?
9.	What are some common pitfalls when working with date/time data in Pandas?
10.	Explain the difference between wide and long data formats and how to convert between them.
11.	How does broadcasting extend to multi-dimensional arrays?
12.	What is the difference between apply(), map(), and applymap() in Pandas?
13.	How does Pandas handle time zones in datetime objects?
14.	Explain how np.vectorize works internally.
15.	What are some best practices for visualizing large datasets?

#### Mini Coding Challenges
1.	Write a function to efficiently flatten a multi-dimensional NumPy array.
2.	Create a Pandas pivot table to summarize data with multiple aggregation functions.
3.	Write a program to detect and handle outliers in a DataFrame column.
4.	Implement a custom Seaborn plot combining multiple plot types.
5.	Optimize a data processing pipeline using vectorized NumPy and Pandas operations.


</details>

<details>
<summary>📙 Week 3: Introduction to Machine Learning workflow using the Iris dataset — explored K-Nearest Neighbors (KNN), train/test split, evaluation metrics (accuracy, classification report, confusion matrix), and cross-validation.</summary>
<br/>

- **Topics Planned**: Data preprocessing, Scikit-learn intro  
- **Status**: 🚧 InProgress   
- **Days Spent**: 2  
- **Time Frame**: Aug 2025 

## Week 3 — Day 1 ✅

### Focus: K-Nearest Neighbors (KNN)

- Learned the **Machine Learning Workflow** (problem definition → data → split → model → evaluation).  
- Explored the **Iris dataset** (features, targets, target names).  
- Performed **train/test split** (80/20) to avoid overfitting.  
- Implemented **KNN classifier** with scikit-learn.  
- Understood key concepts:  
  - `k` = number of neighbors to consult when predicting.  
  - Trade-off between small `k` (more flexible, but noisy) vs. large `k` (smoother, but may miss local patterns).  
  - Why stratification ensures balanced class distribution in splits.  
  - The need for **cross-validation** to select the best `k`.  
- Evaluated model using:  
  - **Accuracy**  
  - **Classification report** (precision, recall, f1-score)  
  - **Confusion matrix**  

✅ **Outcome:** Solid grasp of KNN as a classification algorithm and how to evaluate its performance.

<!--
📌 Model Selection Cheat Sheet

🔹 1. Classification (Predict categories)
	•	Logistic Regression → Simple, interpretable, fast. Works well when classes are linearly separable.
	•	KNN → Easy to understand, good for small datasets. Sensitive to scaling.
	•	Decision Tree → Human-readable flowchart of decisions. Can overfit.
	•	Random Forest / Gradient Boosting → Stronger accuracy, handles complex data.
	•	SVM → Good for clear boundaries, but harder to tune.
	•	Neural Networks → Use only for large/complex datasets.

⸻

🔹 2. Regression (Predict numbers)
	•	Linear Regression → Best when data follows straight-line relations.
	•	Ridge / Lasso → Linear regression with penalties to avoid overfitting.
	•	Decision Tree Regressor → Can capture non-linear patterns.
	•	Random Forest / Gradient Boosting → Stronger for complex data, less interpretable.

⸻

🔹 3. Unsupervised Learning (No labels)
	•	KMeans → Simple, fast clustering. Needs you to pick number of clusters.
	•	DBSCAN / Hierarchical → Detects clusters of varying shapes and sizes.
	•	PCA → Reduce dataset dimensions, keep main information.
	•	t-SNE / UMAP → Great for visualization of high-dimensional data.

-->


</details>

<details>
<summary>📙 Week 4: Coming Soon</summary>
<br/>
- **Status**: 🚧 InProgress 
<br/>

- **Topics Planned**: Data preprocessing, Scikit-learn intro  
- **Status**: 🔜 Planned  
- **Time Frame**: TBD  

</details>
</details>