import os
import pandas as pd
from sklearn.model_selection import train_test_split


from ml.data import process_data, apply_label
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
#load the cencus.csv data
project_path = os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")
data = pd.read_csv(data_path)

# Splitting the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.

train, test = train_test_split(data, 
                               random_state=104,  
                               test_size=0.25,  
                               shuffle=True)


# DO NOT MODIFY
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

# use the ml/data.py: process_data function provided to process the training data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
    )

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# use the ml/model.py: train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# load the model
model = load_model(
    model_path
) 

#use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics

p, r, fb, a = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | Accuracy: {a:.4f} | F1: {fb:.4f}")

# Compute the performance on model slices using the ml/model.py: performance_on_categorical_slice function
# iterate through the categorical features

#Creating empty lists to create dataframe of related information
colList = []
sliceList = []
countList = []
PrecList = []
RecallList = []
F1List = []
AccList = []
# END


for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb, a = performance_on_categorical_slice(
            data = test, 
            column_name = col, 
            slice_value = slicevalue, 
            categorical_features = cat_features, 
            label = 'salary', 
            encoder = encoder, 
            lb = lb, 
            model = model

            # Is the Column_name and Label the same???
            # use test, col and slicevalue as part of the input
            )
        
        # Adding information to seperate lists
        colList.append(col)
        sliceList.append(slicevalue)
        countList.append(count)
        PrecList.append(p)
        RecallList.append(r)
        F1List.append(fb)
        AccList.append(a)
        ## END 

        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | Accuracy: {a:.4f} | F1: {fb:.4f}", file=f)


## Creating a Dataframe of above information to allow for easier review and analysis.
dct = {'ColumnName':colList, 'SliceValue':sliceList, 'SliceCount':countList, 'Precision':PrecList, 'Recall':RecallList, 'Accuracy':AccList, 'F1':F1List}
df = pd.DataFrame(dct)
df.to_csv('slice_output.csv')
## END 
