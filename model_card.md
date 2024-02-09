# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Created by Andrea Bredesen for the Deploying a Machine Learning Model with FastAPI Udacity Project.  The model uses AdaBoostClassifer.
## Intended Use
Based on features "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", and "native-country", this model is used to predict if the individual's annual salary is greater than $50k or equal/less than $50k.
## Training Data
Training Data for this model was found: https://archive.ics.uci.edu/ml/datasets/census+income
train_test_split of 75% was used for training data. 
## Evaluation Data
Testing Data for this model was found: https://archive.ics.uci.edu/ml/datasets/census+income
train_test_split of 25% was used for testing data.
## Metrics
With the following parameters in the AdaBoostClassifier(n_estimators=300, learning_rate=1.75, random_state=47)
the metrics were achieved: Precision: 0.7741 | Recall: 0.6481 | F1: 0.7055
## Ethical Considerations
The model uses slices to evaluate the data, which is saved as "slice_output.txt".  Reviewing this data for strong Precision, Recall, and F1 scores may lead to a discriminatory basis.
## Caveats and Recommendations
A previous Udacity Project, “Supervised Learning: Project: Finding Donors for *CharityML*” required using 3 different models to compare the outcomes using this same data set. During this project, I used LinearSVC, AdaBoostClassifier, and KNeighborClassifer. The metrics comparing these models showed AdaBoostClassifier achieved a better F1 and accuracy score. Using this information, I used the AdaBoostClassifier model.  

