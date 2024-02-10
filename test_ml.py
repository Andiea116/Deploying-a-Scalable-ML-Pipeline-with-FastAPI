import pytest
import os
import pandas as pd
import numpy as np


@pytest.fixture
def data():
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path)
    return data
@pytest.fixture
def myFile():
    project_path = os.getcwd()
    myFile = pd.read_csv('slice_output.csv')
    return myFile

def test_feature_columns_present(data):
    """
    # Test to ensure the Data includes all cat_features columns
    #  cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
    Use of len(cat_features) instead of # allows for additional columns to be added to cat_features and still be found. 
    Not concerned with extra columns, at this time. 
    """
    #get a list of the current DF columns
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
    ImportedColumns = data.columns.tolist()
    i=0
    
    for feature in cat_features:
        if feature in ImportedColumns:
            i+=1  
    assert i==len(cat_features), (f'All catagory feature columns are present in dataset')


def test_data_length(data):
    """
    Test the length of data available is greater than 400.  Model requests n_numbers of 300, which is 75% of the training_testing Split. Therefore, 400.
    """
    assert data.shape[0] > 400, (f'Dataset is greater than 400 items')

def test_myFile(myFile):
    """
    Test the accuracy of the splits is within 95% of original.
    """
    Accuracy = myFile.Accuracy.mean()
    assert Accuracy > .83, (f'Slice Accuracy is within 95% of original testing')





