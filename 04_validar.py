import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

target_val = "mag"
ds_test = "test.csv"

# Desplegar metircas de pesos, intercepcion, r cuadrada
def data_metrics_2(X_01, y_01, modelo):
  m_weights=modelo.coef_
  r_2 =modelo.score(X_01, y_01)
  print(f"pesos: {m_weights}")
  print("intercept {}".format(modelo.intercept_))
  print(f"r2 = {r_2}")
  
# Separar datos en features y target
def split_data(dataframe_name):
  df = pd.read_csv(dataframe_name)
  X_01 = df.drop(target_val,axis=1) 
  y_01 = df[target_val]
  return (X_01,y_01)

# pedecir con el modelo
def data_prediction(model, X_01):
  y_01 = model.predict(X_01)
  return y_01

#funcion de validacion
def test(test_name, filename='mlparams'):
  reg_test = pickle.load(open(filename, 'rb'))
  (X,y)=split_data(val_name)
  y_pred = data_prediction(reg_test, X)
  print('Metricas Validación')
  data_metrics_2(X, y, reg_validate)

test(ds_test)
