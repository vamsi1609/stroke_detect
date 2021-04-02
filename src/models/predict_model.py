import click
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import numpy
import os
import joblib
import yaml
from src.data.make_dataset import read_params
import pandas as pd
import json


def get_metrics(y_pred, y_true):
    '''
    Function to compute the metrics
    '''
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return f1, precision, accuracy, recall


@click.command()
@click.option("--config")
def predict(config):
    ''' Compute f1 score, accuracy, precision and recall
    for the trained model kept in \models of the main directory
    '''

    config = read_params(config)
    score_file = config["reports"]["scores"]
    test_path = config["split_data"]["test_path"]
    target = config["base"]["target_col"]

    test_data = pd.read_csv(test_path, sep=",")
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]
    model_dir = config["model_dir"]

    # Laoding the model
    print("Loading the saved model....")
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("predicting....")    
    y_pred = clf.predict(X_test)

    print("checking the metrics")
    (f1, precision, accuracy, recall)=get_metrics(y_pred, y_test)

    with open(score_file, "w") as f:
        scores={
            'f1': f1,
            'precision': precision,
            'accuracy': accuracy,
            'recall':recall,}
        json.dump(scores, f, indent=4)
    
    print("done")

if __name__ == "__main__":
    predict()