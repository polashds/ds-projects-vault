import os
from pathlib import Path


list_of_files = [

    f"src/__init__.py",
    f"src/model.py",
    f"src/preprocessing.py",
    f"src/schemas.py",
    f"tests/__init__.py",
    f"tests/test_model.py",
    f"tests/test_api.py",

    ".gitignore",
    "README.md",
    "app.py",
    "requirements.txt",
    "train.py",

    "notebooks/exploration.ipynb",
]



for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")