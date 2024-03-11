import os
from pathlib import Path

list_of_files = [
    f"src/__init__.py",
    f"src/components/__init__.py",
    f"src/config/__init__.py",
    f"src/constants/__init__.py",
    f"src/data_access/__init__.py",
    f"src/entity/__init__.py",
    f"src/exceptions/__init__.py",
    f"src/logger/__init__.py",
    f"src/pipeline/__init__.py",
    f"src/utils/__init__.py",
    "setup.py",
    "requirements.txt",
    "schema.yaml",
    "demo.py",
    ".env"
    
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