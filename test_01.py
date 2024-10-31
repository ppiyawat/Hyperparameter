#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:35:22 2024

@author: mons
"""
#  https://www.geeksforgeeks.org/random-forest-hyperparameter-tuning-in-python/


from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 

data = pd.read_csv( 
	"https://raw.githubusercontent.com/lucifertrj/"
	"100DaysOfML/main/Day14%3A%20Logistic_Regression"
	"_Metric_and_practice/heart_disease.csv") 

print(data.head(7) )

print(data['target'].value_counts() )


X = data.drop("target", axis=1) 
y = data['target'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, 
									test_size=0.25, 
									random_state=42) 
X_train.shape, X_test.shape 
