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

### 📑 Week 1 — Question Bank & Coding Challenges

#### Easy Questions
- What is a variable in Python? Give an example.
- How do you print a message to the console?
- What are the basic data types in Python?
- How do you create a list and access its elements?
- What is the difference between a list and a tuple?
- How do you write a comment in Python?
- What is the syntax for defining a function?
- How do you take user input in Python?
- Explain the use of `if` statements with an example.
- How do you write a `for` loop to iterate over a list?

#### Easy Coding Challenges
1. Write a program to print "Hello, World!".
2. Create a list of five numbers and print each number using a loop.
3. Write a function that takes two numbers and returns their sum.
4. Take a user's name as input and greet them.
5. Write a program to check if a number is even or odd.

#### Medium Questions
- What is the difference between `==` and `is` in Python?
- How do you handle errors using try-except blocks?
- Explain list comprehensions with an example.
- What are dictionaries and how do you use them?
- How do you define a class and create an object in Python?
- Explain the difference between mutable and immutable types.
- How do you use `elif` in conditional statements?
- What is the purpose of the `return` statement in functions?
- How do you iterate over keys and values in a dictionary?
- What are Python modules and how do you import them?

#### Medium Coding Challenges
1. Write a function to find the largest number in a list.
2. Create a dictionary mapping student names to their scores and print all names with scores above 80.
3. Write a class `Rectangle` with attributes length and width and a method to calculate area.
4. Use list comprehension to create a list of squares of numbers from 1 to 10.
5. Write a program that counts the number of vowels in a given string.

#### Tricky Questions
- Explain the difference between shallow copy and deep copy.
- How does Python's garbage collection work?
- What are decorators and how are they used?
- Explain the `*args` and `**kwargs` syntax in functions.
- What is the Global Interpreter Lock (GIL) in Python?
- How do you manage memory in Python?
- Explain generators and how they differ from iterators.
- What are lambda functions and where would you use them?
- How does Python handle variable scope?
- What is list slicing and how does it work?

#### Tricky Coding Challenges
1. Write a decorator that logs the execution time of a function.
2. Implement a generator function to yield Fibonacci numbers up to n.
3. Write a function using `*args` and `**kwargs` to accept any number of arguments and print them.
4. Create a class with a class method and a static method and explain their differences.
5. Write a program to flatten a nested list using recursion.

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

### 📑 Week 2 — Question Bank & Coding Challenges

#### Easy Questions
- What is a NumPy array and how is it different from a Python list?
- How do you create a NumPy array?
- What is broadcasting in NumPy?
- How do you select rows and columns in a Pandas DataFrame?
- What is the difference between a Series and a DataFrame in Pandas?
- How do you create a Pandas DataFrame from a dictionary?
- How do you plot a simple bar chart using Matplotlib?
- What is the purpose of `plt.show()` in Matplotlib?
- How do you install and import Seaborn?
- How do you filter a DataFrame based on a condition?

#### Easy Coding Challenges
1. Create a NumPy array of numbers from 1 to 10 and print it.
2. Create a Pandas DataFrame with columns "Name" and "Age" and print it.
3. Plot a bar chart of the number of students in different classes using Matplotlib.
4. Filter a DataFrame to show only rows where the score is above 80.
5. Use Seaborn to create a histogram of a given dataset.

#### Medium Questions
- Explain the difference between `np.split` and `np.array_split`.
- How do you use `groupby` in Pandas and what is it used for?
- What is a Boolean mask in Pandas and how is it applied?
- How do you merge two DataFrames in Pandas?
- What are some common aggregation functions used with `groupby`?
- How do you handle missing data in Pandas?
- Explain the difference between `loc` and `iloc` in Pandas.
- How do you customize the color palette in Seaborn plots?
- How do you save a Matplotlib plot to a file?
- What is the difference between `plt.plot()` and `sns.lineplot()`?

#### Medium Coding Challenges
1. Write a function to compute the mean and standard deviation of a NumPy array.
2. Group a DataFrame by a categorical column and compute the average of a numerical column.
3. Merge two DataFrames on a common key and display the result.
4. Create a scatter plot using Seaborn with custom colors and labels.
5. Write a program to fill missing values in a DataFrame column with the column mean.

#### Tricky Questions
- How does NumPy handle memory differently than Python lists?
- What is the difference between views and copies in NumPy arrays?
- Explain the concept of chaining assignments in Pandas and why it is discouraged.
- How do you optimize Pandas operations for large datasets?
- What are categorical data types in Pandas and why use them?
- How does Seaborn integrate with Matplotlib under the hood?
- Explain how pivot tables work in Pandas.
- How do you create multi-index DataFrames and why are they useful?
- What are some common pitfalls when working with date/time data in Pandas?
- Explain the difference between wide and long data formats and how to convert between them.

#### Tricky Coding Challenges
1. Write a function to efficiently flatten a multi-dimensional NumPy array.
2. Create a Pandas pivot table to summarize data with multiple aggregation functions.
3. Write a program to detect and handle outliers in a DataFrame column.
4. Implement a custom Seaborn plot combining multiple plot types.
5. Optimize a data processing pipeline using vectorized NumPy and Pandas operations.

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