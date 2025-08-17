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