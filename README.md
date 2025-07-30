## 🐍 **Mahi's** Journey into Python and AI Land  
A motivation to learn AI using Python, leading to this commitment.



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

- **Topics Covered**: NumPy arrays, Pandas DataFrames, Matplotlib  
- **Status**: 🚧 In Progress  
- **Days Spent**: 1 (ongoing)  
- **Time Frame**: July 2025  

</details>

<details>
<summary>📙 Week 3: Coming Soon</summary>
<br/>

- **Topics Planned**: Data preprocessing, Scikit-learn intro  
- **Status**: 🔜 Planned  
- **Time Frame**: TBD  

</details>

</details>

#### Note : To preview Readme.md in Vs code  on Mac (Command+Shift+V)