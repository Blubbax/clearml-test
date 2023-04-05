from clearml import Dataset, Task, TaskTypes
import pandas as pd
from pathlib import Path
import os

task = Task.init(
    project_name="Iris",
    task_name="Data Preparation",
    task_type=TaskTypes.data_processing
)

parent_data = Dataset.get(
    dataset_name="iris_raw", 
    dataset_project="Iris"
)

dataset_path = parent_data.get_local_copy()

print(dataset_path)

df = pd.read_csv(Path(dataset_path) / "iris.csv")
df['sepal_length_squared'] = df["sepal_length"] ** 2
df['sepal_width_squared'] = df["sepal_width"] ** 2
df['petal_length_squared'] = df["petal_length"] ** 2
df['petal_width_squared'] = df["petal_width"] ** 2

df['class'] = df['class'].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

data_dir = Path(os.path.abspath('')) / 'iris.csv'
df.to_csv(data_dir, index=False)

dataset = Dataset.create(
  dataset_name="iris_processed",
  dataset_project="Iris",
  parent_datasets=[parent_data.id]
)

dataset.add_files(data_dir)
dataset.upload()
dataset.finalize()

os.remove(data_dir)
