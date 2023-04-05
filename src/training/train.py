import logging
from pandas import read_csv
from clearml import Dataset, Task, TaskTypes, Logger, OutputModel
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from pathlib import Path
import os
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def generate_datasets(X, Y, train_index, test_index):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = Y.iloc[train_index]
    Y_test = Y.iloc[test_index]
    return X_train, X_test, Y_train, Y_test



task = Task.init(
    project_name="Iris",
    task_name="Train",
    task_type=TaskTypes.training
)
# task.execute_remotely(queue_name="default")

logger = task.get_logger()

params = {
    'columns': 2,
    'folds': 10,
    'score': 'accuracy'
}

task.connect(params)

parent_data = Dataset.get(
    dataset_name="iris_processed", 
    dataset_project="Iris"
)

dataset_path = parent_data.get_local_copy()
print(dataset_path)
df = pd.read_csv(Path(dataset_path) / "iris.csv", encoding= 'unicode_escape')

Y = df[['class']].copy()
X = df.drop(columns=['class'])

kf = KFold(n_splits=params['folds'], shuffle=True)

best_score = -10

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test, Y_train, Y_test = generate_datasets(X, Y, train_index, test_index)

    model = LogisticRegression(solver='liblinear')
    rfe = RFE(model, n_features_to_select=params['columns'])
    fit = rfe.fit(X_train, Y_train)
    prediction = rfe.predict(X_test)

    accuracy = accuracy_score(Y_test, prediction)
    f1 = f1_score(Y_test, prediction, average='weighted')

    logger.report_scalar(
        "Accuracy", "Accuracy", iteration=i, value=accuracy
    )
    logger.report_scalar(
        "F1", "F1", iteration=i, value=f1
    )

    if params['score'] == 'f1':
        if f1 > best_score:
            best_score = f1
            best_model = fit
    else:
        if accuracy > best_score:
            best_score = accuracy
            best_model = fit

module_path = Path(os.path.abspath(os.getcwd()))
model_path = module_path / "models" / ("iris_"+str(task.id)+".onnx")

logger.report_text(str(best_model.get_feature_names_out()))

initial_type = [('float_input', FloatTensorType([None, 8]))]

onnx = convert_sklearn(best_model, initial_types=initial_type)
with open(os.path.abspath(model_path), "wb") as f:
    f.write(onnx.SerializeToString())

output_model = OutputModel(task=task)
output_model.update_weights(register_uri=os.path.abspath(model_path))

