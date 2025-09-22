> [🔼 README](../../README.md)

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
