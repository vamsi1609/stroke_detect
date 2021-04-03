# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def read_params(config_path):
    '''
    takes the config_path as input and 
    returns the loaded yaml file 
    '''
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def split_n_save(config_path):
    '''
    Function will split the data into train and test set
    and will save it in data\processed folder
    '''
    config = read_params(config_path)
    # Fetching the configurations
    train_path = config["split_data"]["train_path"]
    test_path = config["split_data"]["test_path"]
    inter_path = config["interim_data"]["source"]
    test_split = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]
    #Processing and saving
    print(train_path)
    print(test_path)
    df = pd.read_csv(inter_path, sep=",")
    train, test = train_test_split(df,
                                   test_size=test_split,
                                   random_state=random_state)
    train.to_csv(train_path, sep=",", index=False)
    test.to_csv(test_path, sep=",", index=False)


@click.command()
@click.option('--config', default="params.yaml")
def main(config):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    split_n_save(config)
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()