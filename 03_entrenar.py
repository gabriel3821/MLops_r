import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

def data_metrics(yP, yL):
    print(accuracy_score(yP,yL))
    print(confusion_matrix(yP,yL))
    print(classification_report(yP,yL))

def split_data(dataframe_name):
    df = pd.read_csv(dataframe_name)
    X_01 = df.drop('test_result',axis=1) 
    y_01 = df['test_result']
    return (X_01,y_01)

def data_save(log_model, filename='mlparams'):
    pickle.dump(log_model, open(filename, 'wb'))
    print('Parametros escritos en archivo mlparams')

def data_prediction(log_model, X_01):
    y_01 = log_model.predict(X_01)
    return y_01

def training(train_name):
    (X,y)=split_data(train_name)
    log_train = LogisticRegression()
    log_train.fit(X,y)
    y_pred = data_prediction(log_train, X) 
    data_save(log_train)
    print('Metricas Entrenamiento')
    data_metrics (y_pred,y)

def validation(test_name, filename='mlparams'):
    log_test = pickle.load(open(filename, 'rb'))
    (X,y)=split_data(test_name)
    y_pred = data_prediction(log_test, X)
    print('Metricas Validación')
    data_metrics (y_pred,y)

training('train_01.csv')   
validation('train_02.csv')
