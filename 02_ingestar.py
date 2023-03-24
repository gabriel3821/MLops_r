#carga
import pandas as pd
from sklearn.model_selection import train_test_split

def save_data(set_name,dataset):
	dataset.to_csv(set_name + ".csv",index=False)
	print("Dataset "+ set_name +".csv listo ...")

def fix_data(A,B):
  ds= pd.DataFrame()
  ds= A
  ds['test_result']=B
  return ds

def split_data(dataframe_name, validation=False, ss=0.1):
  datasets= pd.DataFrame()
  df = pd.read_csv(dataframe_name)
  X = df.drop('test_result',axis=1) 
  y = df['test_result']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ss, random_state=101)
  if validation == False: 
    setname_01='train'
    setname_02='test'
  else :
    setname_01='train_01'
    setname_02='train_02'
  datasets=fix_data(X_train,y_train)
  save_data(setname_01,datasets)
  datasets=fix_data(X_test,y_test)
  save_data(setname_02,datasets)

split_data('quakes.csv', validation = False, ss = 0.2)
split_data('train.csv', validation = True, ss = 0.1)
