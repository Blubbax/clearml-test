from clearml import Dataset, Task, TaskTypes
import mysql.connector
import pandas as pd
from pathlib import Path
import os

task = Task.init(
    project_name="Iris",
    task_name="Raw Data Collection",
    task_type=TaskTypes.data_processing
)

task.execute_remotely(queue_name="default")
print("Start task")

params = {
    'db_username': 'irisuser',
    'db_password': 'ABC123abc.'
}

task.connect(params)

print("Connect to DB")
mydb = mysql.connector.connect(
  host="192.168.178.33",
  user=params['db_username'],
  password=params['db_password'],
  database="iris",
  port="3307"
)

print("Query data")
mycursor = mydb.cursor()
mycursor.execute("SELECT * FROM iris_data")
myresult = mycursor.fetchall()
df = pd.DataFrame(myresult)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class' ]

data_dir = Path(os.path.abspath('')) / 'iris.csv'

print("Store temporarily")
df.to_csv(data_dir, index=False)

print("Create and store dataset to ClearML")
dataset = Dataset.create(
  dataset_name="iris_raw",
  dataset_project="Iris"
)

dataset.add_files(data_dir)
dataset.upload()
dataset.finalize()

os.remove(data_dir)

print("Task ready")