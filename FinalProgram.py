#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pickle
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, RandomizedSearchCV
from sklearn import preprocessing
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, mean_squared_error


# In[9]:


#Get some global values
current_dir = os.getcwd()
training_path = current_dir.replace('Program','trained_models') 
data_path = current_dir.replace('Program','data')

os.chdir(data_path)
mean = np.loadtxt('mean.csv',delimiter=',')
std = np.loadtxt('std.csv',delimiter=',')
os.chdir(current_dir)


# In[22]:


def get_all_models():
    global training_path
    models_path = training_path
    models = []
    for file in os.listdir(models_path):
        filepath = models_path + '\\' + file
        with open(filepath,'rb') as f:
            models.append(pickle.load(f))
    return models

def get_custom_input():
    #------------------------------------For predicting attrition on a new employee-----------------------------------------------#

    global mean
    global std
    attributes = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance',
           'JobInvolvement', 'PerformanceRating', 'BusinessTravel',
           'DistanceFromHome', 'Education', 'EducationField', 'Gender',
           'MaritalStatus', 'MonthlyIncome', 'TotalWorkingYears',
           'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',
           'in_avg', 'out_avg', 'avg_work_day', 'num_day_off',
           'Age']
    x_test = []
    print("I hope you've read the data dictionary before inputing the values")
    print("If you do not have info about a particular part, just type 'NA' and we will use average values from our data")
    count = 0
    for atr in attributes:
        if atr in ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance','JobInvolvement','PerformanceRating']:
            status = 'illegal'
            while status == 'illegal':
                temp = input('Enter {} on a scale of 1 to 4 with 1 being lowest'.format(atr))
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                if temp < 1.0 or temp > 4.0:
                    print('Enter a valid rating')
                else:
                    x_test.append(temp)
                    count += 1
                    status = 'legal'

            continue

        if atr == 'BusinessTravel':
            status = 'illegal'
            while status == 'illegal':
                temp = input('Enter type of Business Travel (0,1,2 if travelling never, frequently, rarely respectively)')
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                if temp < 0 or temp > 2:
                    print('Enter valid travel type')
                else:
                    status = 'legal'
                    x_test.append(temp)
                    count += 1
            continue

        if atr == 'DistanceFromHome':
            temp = input('Enter distance to office from home in km')
            if temp == 'NA':
                x_test.append(mean[count])
                count += 1
                continue
            temp = float(temp)
            x_test.append(temp)
            count += 1
            continue

        if atr == 'Education':
            status = 'illegal'
            while status == 'illegal':
                temp = input('Enter education level of employee')
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                if temp < 1 or temp > 5:
                    print('Enter a valid education level, 1 for below college level, 5 for Doctoral level. See Dictionary for other details')
                else:
                    status = 'legal'
                    x_test.append(temp)
                    count += 1
            continue

        if atr == 'EducationField':
            status = 'illegal'
            while status == 'illegal':
                temp = input('Enter field of education (refer to the dictionary)')
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                if temp < 0 or temp > 5:
                    print('Enter a valid value')
                else:
                    status = 'legal'
                x_test.append(temp)
                count += 1
            continue

        if atr == 'Gender':
            status = 'illegal'
            while status == 'illegal':
                temp = input('Enter gender (0 for female, 1 for male, or in between 0 to 1 as per the employee preference) ')
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                if temp  < 0 or temp > 1:
                    print('Please enter between 0 and 1')
                else:
                    status = 'legal'
                x_test.append(temp)
                count += 1
            continue

        if atr == 'MaritalStatus':
            status = 'illegal'
            while status == 'illegal':
                temp = input('Enter marital status (1 if married, 2 if unmarried, 0 if divorced)')
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                if temp < 0 or temp > 2:
                    print('Enter correct status')
                else:
                    status = 'legal'
                x_test.append(temp)
                count += 1
            continue



        if atr == 'MonthlyIncome':
            status = 'illegal'
            while status == 'illegal':
                temp = input('Enter Monthly income (in dollars)')
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                status = 'legal'
                x_test.append(temp)
                count += 1
            continue


        if atr == 'TotalWorkingYears':
            status = 'illegal'
            while status == 'illegal':
                temp = input('Enter total working years')
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                status = 'legal'
                x_test.append(temp)
                count += 1
            continue

        if atr == 'TrainingTimesLastYear':
            status = 'illegal'
            while status == 'illegal':
                temp = input('Enter total times employee took training last year')
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                status = 'legal'
                x_test.append(temp)
                count += 1
            continue

        if atr == 'YearsAtCompany':
            status = 'illegal'
            while status == 'illegal':
                temp = input('Enter total years employee is at your company')
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                status = 'legal'
                x_test.append(temp)
                count += 1
            continue

        if atr == 'YearsSinceLastPromotion':
            status = 'illegal'
            while status == 'illegal':
                temp = input("Enter total years since the employee's last promotion")
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                status = 'legal'
                x_test.append(temp)
                count += 1
            continue

        if atr == 'in_avg':
            status = 'illegal'
            while status == 'illegal':
                temp = input("Enter time when employee generally comes to office (in 24 hour format, for eg: 9 for 9 AM)")
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                status = 'legal'
                x_test.append(temp)
                count += 1
            continue

        if atr == 'out_avg':
            status = 'illegal'
            while status == 'illegal':
                temp = input("Enter time when employee generally leaves office (in 24 hour format, for eg: 9 for 9 AM)")
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                status = 'legal'
                x_test.append(temp)
                count += 1
            continue

        if atr == 'avg_work_day':
            status = 'illegal'
            while status == 'illegal':
                temp = input("Enter number of hours, employee spends at the office per day")
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                status = 'legal'
                x_test.append(temp)
                count += 1
            continue

        if atr == 'avg_work_day':
            status = 'illegal'
            while status == 'illegal':
                temp = input("Enter number of hours, employee spends at the office per day")
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                if temp > 24 or temp < 0:
                    print('Employee cannot spend more than 24 hours or negative hours everyday')
                else:
                    status = 'legal'
                    x_test.append(temp)
                    count += 1
            continue

        if atr == 'num_day_off':
            status = 'illegal'
            while status == 'illegal':
                temp = input("Enter number of days employee takes an off")
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    continue
                temp = float(temp)
                if temp < 0:
                    print("Is this an employee who comes in to work on holidays? If yes, give us no. of days he is absent on working days and doesn't make up for them")
                else:
                    status = 'legal'
                    x_test.append(temp)
                    count += 1
            continue

        if atr == 'Age':
            status = 'illegal'
            while status == 'illegal':
                temp = input('Enter age in years {}'.format(atr))
                if temp == 'NA':
                    x_test.append(mean[count])
                    count += 1
                    x_test.append(mean[count])
                    count += 1
                    x_test.append(mean[count])
                    count += 1
                    x_test.append(mean[count])
                    count += 1
                    status = 'legal'
                    break
                age = float(temp)
                if (age > 17.958 and age < 28.5):
                    x_test.append(1)
                    x_test.append(0)
                    x_test.append(0)
                    x_test.append(0)
                    status = 'legal'
                elif (age > 28.5 and age < 39.0):
                    x_test.append(0)
                    x_test.append(1)
                    x_test.append(0)
                    x_test.append(0)
                    status = 'legal'
                elif (age > 39.0 and age < 49.5):
                    x_test.append(0)
                    x_test.append(0)
                    x_test.append(1)
                    x_test.append(0)
                    status = 'legal'
                elif (age > 49.5 and age < 60):
                    x_test.append(0)
                    x_test.append(0)
                    x_test.append(0)
                    x_test.append(1)
                    status = 'legal'
                else:
                    print('Illegal Age, enter between 18 to 60')
            continue 

    x_test = np.array(x_test,dtype=float)
    x = (x_test-mean)/std
    return x

def predict():
    #Makes use of all models and gives an overall estimate
    x = get_custom_input()
    models = get_all_models()
    x = np.array(x)
    y_predicts = []
    for model in models:
        y_predicts.append(model.predict(x.reshape((1,24))))
    y = np.array(y_predicts)
    counts = np.sum(y)
    print('Our {} out of {} models predict employee will exit'.format(counts,len(y)))
    return


# In[23]:


predict()


# In[ ]:




