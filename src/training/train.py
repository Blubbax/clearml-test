try:
    import joblib
except ImportError:
    from sklearn.externals import joblib
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
from model.model import ModelLogReg




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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

kf = KFold(n_splits=params['folds'], shuffle=True)

model = ModelLogReg(params['columns']).get_model()
model.fit(X_train, y_train)

# joblib.dump(model, 'model.pkl', compress=True)

results = cross_val_score(model, X, Y, cv=kf, scoring=params['score'])

logger.report_scalar(
    params['score'], params['score'], iteration=0, value=results.mean()
)

module_path = Path(os.path.abspath(os.getcwd()))
model_path = module_path / "models" / ("iris_"+str(task.id)+".onnx")

# logger.report_text(str(best_model.get_feature_names_out()))

initial_type = [('float_input', FloatTensorType([None, 8]))]

# Convert to ONNX
onnx = convert_sklearn(model, initial_types=initial_type)
with open(os.path.abspath(model_path), "wb") as f:
    f.write(onnx.SerializeToString())

output_model = OutputModel(task=task)
output_model.update_weights(register_uri=os.path.abspath(model_path))

