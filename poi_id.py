#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    data = featureFormat(data_dict, features_list)
    
### Task 2: Remove outliers
    del data_dict["TOTAL"]
    

### Task 3: Create new feature(s)
    def computeFraction( poi_messages, all_messages ):    
        if all_messages=='NaN' or poi_messages=='NaN':
            fraction = 0
        else:
            fraction = float(poi_messages)/float(all_messages)
        return fraction


    for name in data_dict:
        #making two new features
        from_poi_to_this_person = data_dict[name]["from_poi_to_this_person"]
        to_messages = data_dict[name]["to_messages"]
        data_dict[name]["fraction_from_poi"] = computeFraction( from_poi_to_this_person, to_messages )

        from_this_person_to_poi = data_dict[name]["from_this_person_to_poi"]
        from_messages = data_dict[name]["from_messages"]    
        data_dict[name]["fraction_to_poi"] = computeFraction( from_this_person_to_poi, from_messages )
    




features_list = ['poi', 'bonus','exercised_stock_options']
# Result, Precision: 0.4747Recall: 0.40850
# Importance {'bonus': 0.46998094273401092, 'exercised_stock_options': 0.53001905726598908}

# features_list = ['poi','salary', 'deferral_payments', 'total_payments', \
#                 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',\
#                 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', \
#                 'long_term_incentive', 'restricted_stock', 'director_fees',\
#                 'to_messages', 'from_poi_to_this_person', 'from_messages', \
#                 'from_this_person_to_poi', 'shared_receipt_with_poi',\
#                  'fraction_from_poi','fraction_to_poi']
#Result, Precision: 0.4809 Recall: 0.10100


# features_list = ['poi', 'bonus','exercised_stock_options', 'from_this_person_to_poi','from_poi_to_this_person']
# Result, Precision: 0.4473 Recall: 0.26550

# features_list = ['poi', 'bonus','exercised_stock_options','total_stock_value']
#Result, Precision: 0.5558 Recall: 0.38600

# features_list = ['poi', 'bonus','exercised_stock_options','total_stock_value','expenses']
#Result, Precision: 0.6017 Recall: 0.34000


### Task 4: Try a varity of classifiers
### Store to my_dataset for easy export below.
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


##########################
#SVM
##########################
#scaling
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
features_minmax = min_max_scaler.fit_transform(features)

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

# clf_params= {
#                        'clf__C': [1e-5, 1e-2, 1e-1, 1, 10, 1e2, 1e5],
#                        'clf__gamma': [0.0],
#                        'clf__kernel': ['linear', 'poly', 'rbf'],
#                        'clf__tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  
#                        'clf__class_weight': [{True: 12, False: 1},
#                                                {True: 10, False: 1},
#                                                {True: 8, False: 1},
#                                                {True: 15, False: 1},
#                                                {True: 4, False: 1},
#                                                'auto', None]
#                       }

# #For this Pipeline:
# pipe = Pipeline(steps=[('minmaxer', MinMaxScaler()), ('clf', SVC())])
# cv = cross_validation.StratifiedShuffleSplit(labels,n_iter = 50,random_state = 42)
# a_grid_search = GridSearchCV(pipe, param_grid = clf_params,cv = cv, scoring = 'recall')
# a_grid_search.fit(features,labels)

# # pick a winner
# best_clf = a_grid_search.best_estimator_
# print best_clf

#this is the parametor I got 
# (C=1e-05, cache_size=200, class_weight={False: 1, True: 12}, coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=0.0, kernel='linear',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.1, verbose=False)

#Making SVM model
# clf=SVC(C=1e-05, cache_size=200, class_weight={False: 1, True: 12}, coef0=0.0,
#   decision_function_shape=None, degree=3, gamma=0.0, kernel='linear',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.1, verbose=False).fit(features_minmax, labels)

#Result
#Precision: 0.20987  Recall: 0.92900 




##########################
#Random Forest
##########################

from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier(n_estimators=50,max_features=2)
clf.fit(features,labels)

########################


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)