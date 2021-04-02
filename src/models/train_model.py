import click
from sklearn.ensemble import RandomForestClassifier
import numpy
import os
import joblib
import yaml
from src.data.make_dataset import read_params
import pandas as pd
import json


@click.command()
@click.option("--config")
def train(config):
    config = read_params(config)

    # Parameters
    train_path = config["split_data"]["train_path"]
    target = config["base"]["target_col"]
    n_estimators = config["estimators"]["n_estimators"]
    model_dir = config["model_dir"]
    random_state = config["base"]["random_state"]

    # Train and test data
    print("Creating the data.......")
    train_data = pd.read_csv(train_path, sep=",")
    

    # Train and Test data split
    X_train = train_data.drop(target, axis=1)
    
    y_train = train_data[target]
    

    # Training the classifier
    print("Training the model.............")
    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    print("Saving the model........")
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(clf, model_path)

    print("Adding reports........")
    params_file = config["reports"]["params"]

    with open(params_file, "w") as f:
        params = {
            'n_estimators': n_estimators,
        }
        json.dump(params, f, indent=4)


if __name__ == "__main__":
    train()
