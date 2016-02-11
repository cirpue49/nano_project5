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
    # max_salary =  sorted(data,key=lambda l:l[1],reverse=True)[0][1]
    # second_salary=sorted(data,key=lambda l:l[1],reverse=True)[1][1]
    # min_salary = sorted(data,key=lambda l:l[1],reverse=True)[-1][1]
    # for key in data_dict:
    #   if data_dict[key]['salary']==max_salary or data_dict[key]['salary']==min_salary or data_dict[key]['salary']==second_salary:
    #     print data_dict[key],key
    # print data_dict['TOTAL']
    # print len(data_dict)
### Task 2: Remove outliers
    del data_dict["TOTAL"]
    # print len(data_dict)

### Task 3: Create new feature(s)
    def computeFraction( poi_messages, all_messages ):    
        if all_messages=='NaN' or poi_messages=='NaN':
            fraction = 0
        else:
            fraction = float(poi_messages)/float(all_messages)
        return fraction

    for name in data_dict:

        from_poi_to_this_person = data_dict[name]["from_poi_to_this_person"]
        to_messages = data_dict[name]["to_messages"]
        data_dict[name]["fraction_from_poi"] = computeFraction( from_poi_to_this_person, to_messages )


        from_this_person_to_poi = data_dict[name]["from_this_person_to_poi"]
        from_messages = data_dict[name]["from_messages"]    
        data_dict[name]["fraction_to_poi"] = computeFraction( from_this_person_to_poi, from_messages )
    # print data_dict['GLISAN JR BEN F']['fraction_from_poi']
    # print data_dict['GLISAN JR BEN F']['fraction_to_poi']
### Store to my_dataset for easy export below.
my_dataset = data_dict


# for i in my_dataset
### Extract features and labels from dataset for local testing
features_list = ['poi','salary','fraction_from_poi','fraction_to_poi', 'shared_receipt_with_poi']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)



# from sklearn.svm import SVC
# clf=SVC(kernel='rbf',C=10000)
# clf.fit(features_train,labels_train)
# pred=clf.predict(features_test)

from sklearn.metrics import accuracy_score, precision_score

# print accuracy_score(labels_test,pred)
# print clf.predict(features_test)
# print precision_score(labels_test,pred)


### Task 4: Try a varity of classifiers
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# clf.fit(features_train,labels_train)
# pred=clf.predict(features_test)

# print accuracy_score(labels_test, pred)
# print precision_score(labels_test,pred)


from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier(n_estimators=50,max_features=2)
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)

# print len(labels_test)
# print clf.feature_importances_
# print accuracy_score(labels_test, pred)
# print precision_score(labels_test,pred)

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)