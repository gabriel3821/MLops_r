#carga
import pandas as pd
from sklearn.model_selection import train_test_split

# Definicion de valores
target_val = "mag"
dataset_name = "quakes.csv"
dataset_name_01 = "train.csv"

# Salvar el dataset
def save_data(set_name,dataset):
  dataset.to_csv(set_name + ".csv", index=False, columns=['lat','long','depth','stations','mag'])
  print("Dataset "+ set_name +".csv listo ...")

# Junta los datos en un dataset
def fix_data(A,B,dataset):
  ds= pd.DataFrame(data = A)
  ds[target_val]=B
  print(ds.head(5))
  save_data(dataset,ds)

# Dividir datos en 3 datasets
def split_data(dataframe_name, validation=False, ss=0.1):
  df = pd.read_csv(dataframe_name)
  X = df.drop(target_val,axis=1) 
  y = df[target_val]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ss, random_state=42)
  if validation == False: 
    setname_01='train'
    setname_02='test'
  else :
    setname_01='train'
    setname_02='validate'
  fix_data(X_train,y_train,setname_01)
  fix_data(X_test,y_test,setname_02)

# llamada de funciones
split_data(dataset_name, validation = False, ss = 0.2)
split_data(dataset_name_01, validation = True, ss = 0.1)
