#inferir
from flask import Flask
from flask import request
from sklearn.linear_model import LogisticRegression
import pickle
import logging
import sys
import numpy as np
 
print(__name__)
app = Flask(__name__)

filename = "mlparams"
api_mlparams = pickle.load(open(filename, 'rb'))

logging.info(api_mlparams)

@app.route('/infer')
def infer():  
    reqX_1 = request.args.get('x_1')
    reqX_2 = request.args.get('x_2')
    x_1 = float(reqX_1)
    x_2 = float(reqX_2)
    print((x_1,type(x_1)),file=sys.stderr)
    print((x_2,type(x_2)),file=sys.stderr)
    X_in = np.ndarray(shape=(1,2),buffer=np.array([x_1, x_2]) )
    return {'y':api_mlparams.predict(X_in).item()}

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False)
