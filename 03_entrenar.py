import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Definicion de valores
target_val = "mag"
ds_train = "quakes.csv"
ds_validate = "validate.csv"

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

# persistir al modelo
def data_WR(model,up=False,filename='mlparams'):
  pickle.dump(model, open(filename, 'wb'))
  if up==False:
    print('Parametros escritos en archivo mlparams')
  else:
    print('Parametros actualizados en archivo mlparams')

# pedecir con el modelo
def data_prediction(model, X_01):
  y_01 = model.predict(X_01)
  return y_01

#Funcion de Entrenamineto
def training(train_name):
  (X,y)=split_data(train_name)
  reg_train = LinearRegression().fit(X,y)
  print('Metricas Entrenamiento')
  data_metrics_2(X, y, reg_train)
  
  quakes_weights=reg_train.coef_
  y_pred = data_prediction(reg_train, X.iloc[1:10]) 
  print("comprobar con un dato si se cumple b0 + w0*x0 + w1*x1 + w2*x2 + w3*x3")
  print(reg_train.intercept_+X.iloc[9].lat*quakes_weights[0]+X.iloc[9].long*quakes_weights[1] + X.iloc[9].depth*quakes_weights[2]+X.iloc[9].stations*quakes_weights[3])

  y_quakes_estimation = data_prediction(reg_train, X) 
  print(y_quakes_estimation[0:5])
  data_WR(reg_train,False)

#funcion de validacion
def validation(val_name, filename='mlparams'):
  reg_train = pickle.load(open(filename, 'rb'))
  (X,y)=split_data(val_name)
  reg_validate = reg_train.fit(X, y)
  print('Metricas Validación')
  data_metrics_2(X, y, reg_validate)

  y_pred = data_prediction(reg_validate, X)
  data_WR(reg_validate,True)

training(ds_train)   
validation(ds_validate)
