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
    reqX_3 = request.args.get('x_3')
    reqX_4 = request.args.get('x_4')
    x_1 = float(reqX_1)
    x_2 = float(reqX_2)
    x_3 = float(reqX_3)
    x_4 = float(reqX_4)
    print((x_1,type(x_1)),file=sys.stderr)
    print((x_2,type(x_2)),file=sys.stderr)
    print((x_3,type(x_3)),file=sys.stderr)
    print((x_4,type(x_4)),file=sys.stderr)
    X_in = np.ndarray(shape=(1,4),buffer=np.array([x_1, x_2, x_3, x_4]) )
    return {'y':api_mlparams.predict(X_in).item()}

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False)
