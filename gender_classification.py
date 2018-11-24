import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import xgboost

def CountVectorizer_array(corpus):
    vectorizer = CountVectorizer(max_df=0.9,stop_words='english')
    x=vectorizer.fit_transform(corpus)
    return x.toarray()

if __name__=="__main__":

    #Load the data into a a pandas dataframe and create the count vectors
    df=pd.read_csv('sample.csv',header=None,names=['sku','gender','title'])
    df=df.fillna('')
    corpus=df['title'].tolist()
    Count_vectors=CountVectorizer_array(corpus)
    df['vector']=Count_vectors.tolist()
    train, test = train_test_split(df, test_size = 0.2)

    #Prepare the X_train,X_test,y_train,y_test
    y_train=train['gender']
    X_train=np.array(train['vector'].tolist())
    X_test=np.array(test['vector'].tolist())
    y_test=test['gender']

    #Create the xgboost model
    model=xgboost.XGBClassifier()
    model.fit(X_train,y_train)

    #predict on the test data set 
    y_pred=model.predict(X_test)
    print zip(test['title'].tolist(),y_pred)

    #Compute the accuracy
    xx=[True for x,y in enumerate(y_test) if y==y_pred[x]]
    accuracy=float(len(xx))/float(len(y_test))
    print accuracy
