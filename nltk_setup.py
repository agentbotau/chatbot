import nltk

# Download the 'punkt' data if it's not already downloaded
nltk.download('punkt')

# Check if the 'punkt' tokenizer data exists
from nltk.data import find
try:
    find('tokenizers/punkt')
    print("Punkt tokenizer data is available.")
except LookupError:
    print("Punkt tokenizer data is not available.")

import os

# Define the path
punkt_path = os.path.expanduser('~\\AppData\\Roaming\\nltk_data\\tokenizers\\punkt')
py3_path = os.path.join(punkt_path, 'PY3')
file_path = os.path.join(py3_path, 'PY3_tab')

# Create the directory if it does not exist
os.makedirs(py3_path, exist_ok=True)

# Create the empty file if it does not exist
if not os.path.isfile(file_path):
    open(file_path, 'a').close()

print(f"File created or already exists at {file_path}")

