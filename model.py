import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

raw_mail_data = pd.read_csv("F:\\flask pracs\\csvfiles\\mail_data.csv")
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

X = mail_data['Message']
Y = mail_data['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()
model.fit(X_train_features,Y_train)
#pickle.dump(model, open('Lregressor.pkl','wb'))

#laoding pickle back
#model = pickle.load(open('model.pkl','rb'))

#input_mail = ["Aft i finish my lunch then i go str down lor. Ard 3 smth lor. U finish ur lunch already?"]
#converting text to feature vectors
#input_data_features = feature_extraction.transform(input_mail)
#prediction = model.predict(input_data_features)
