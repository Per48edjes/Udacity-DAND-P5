#!/usr/bin/python

'''
IMPORTANT NOTE:

    This script only includes codes for the classifier chosen for submission. 
    
    For an overview of the end-to-end investigation see the associated Jupyter
    Notebook, which includes a comparison of several implemented algorithms and
    the assignment's associated questions.

'''



### Import all dependencies and set up environment
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# SKLearn packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Data wrangling, outlier removal
for x in ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']:
    data_dict.pop(x, None)

my_dataset = data_dict

# Features lists
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_label = ['poi']

# Remove NaNs
for person, features in my_dataset.iteritems():
    
    # Replace 'NaN's for financial features
    for fin_feature in financial_features:
        if features[fin_feature] == 'NaN':
            features[fin_feature] = 0


### Create custom features 
for person in my_dataset:
    
    # POI sent incidence
    try:
        my_dataset[person]['poi_sent_incidence'] = my_dataset[person]['from_this_person_to_poi'] / float(my_dataset[person]['from_messages'])
    except:
        my_dataset[person]['poi_sent_incidence'] = 'NaN'
    
    # Direct POI involvement factor
    try:
        my_dataset[person]['dpoif'] = my_dataset[person]['from_poi_to_this_person'] / float(my_dataset[person]['shared_receipt_with_poi'])
    except:
        my_dataset[person]['dpoif'] = 'NaN'
        
    # Total compensation
    try:
        my_dataset[person]['total_comp'] = my_dataset[person]['total_payments'] + float(my_dataset[person]['total_stock_value'])
    except:
        my_dataset[person]['total_comp'] = 'NaN'
    
    # Stock comp proportion
    try:
        my_dataset[person]['stock_comp_proportion'] = my_dataset[person]['total_stock_value'] / float(my_dataset[person]['total_comp'])
    except:
        my_dataset[person]['stock_comp_proportion'] = 'NaN'


### Data import, feature selection
feature_list = poi_label + ['poi_sent_incidence', 'total_stock_value', 'shared_receipt_with_poi'] 

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)


### Set up pipeline
pipe = Pipeline([('dtc', DecisionTreeClassifier(presort = True))])


### Optimize parameters
params = {'dtc__min_samples_split': range(2,11), 
          'dtc__max_depth': range(2,11), 
          'dtc__criterion': ['entropy', 'gini']}

print "Fitting classifier!"

### Fit classifier
grid_search = GridSearchCV(pipe, cv = cv, param_grid = params, scoring = 'f1', verbose = 1, n_jobs = 3)
gs_fit = grid_search.fit(features, labels)
print gs_fit.best_params_
clf = gs_fit.best_estimator_

print "Done fitting classifier!"

### Task 6: Dump your classifier, dataset, and features_list 
dump_classifier_and_data(clf, my_dataset, feature_list)
