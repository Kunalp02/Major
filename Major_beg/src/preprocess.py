import pandas as pd

# Load the dataset
dataset_path = "../data/dataset.csv"
df = pd.read_csv(dataset_path)

# Data Preprocessing
# Clean and preprocess text data
def preprocess_text(text):
    text = text.lower()
    return text

# Apply preprocessing to relevant columns
text_columns = ['Paper Title', 'Key Words', 'Abstract', 'Conclusion', 'Summarization']
for col in text_columns:
    df[col] = df[col].apply(lambda x: preprocess_text(x) if pd.notna(x) else x)

# Data Splitting
# Divide the dataset into training, validation, and test sets
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_size = int(len(df) * train_ratio)
val_size = int(len(df) * val_ratio)

train_data = df[:train_size]
val_data = df[train_size:train_size + val_size]
test_data = df[train_size + val_size:]

# Print some basic statistics
print("Total number of papers:", len(df))
print("Number of training papers:", len(train_data))
print("Number of validation papers:", len(val_data))
print("Number of test papers:", len(test_data))
