from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = load_model('titanic_survivor_model.h5')
scaler = joblib.load(r"scaler")

# Test to control that predictions done with model match those processed by this code:

pers_1 = [[1, 2, 1, 2, 151, 1]]
pers_1 = scaler.transform(pers_1)
res_1 = model.predict(pers_1)
print(f'Allison, Miss. Helen Loraine - died : expected=[[0.8820064]] --> actual={res_1}')

pers_2 = [[3, 31, 1, 0, 18, 0]]
pers_2 = scaler.transform(pers_2)
res_2 = model.predict(pers_2)
print(f'Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele) - died : expected=[[0.3935197]] --> actual={res_2}')

pers_3 = [[3, 14, 0, 0, 7.8, 0]]
pers_3 = scaler.transform(pers_3)
res_3 = model.predict(pers_3)
print(f'Vestrom, Miss. Hulda Amanda Adolfina - died : expected=[[0.8711046]] --> actual={res_3}')

pers_4 = [[3, 41, 0, 5, 39.6, 0]]
pers_4 = scaler.transform(pers_4)
res_4 = model.predict(pers_4)
print(f'Panula, Mrs. Juha (Maria Emilia Ojala) - died : expected=[[0.25682092]] --> actual={res_4}')

def req(column):
  return request.form.get(column)

@app.route('/',methods=['post','get']) 
def predict():
  try:
  
    results = [req('pclass'), req('age'), req('sibsp'), req('parch'), req('fare'), req('bcabin')]

    for res in results:
      if res == None:
        return render_template('index.html', result = 'Missing input')
      else:
        arr_2d = np.array([results], dtype=float)

      arr_2d = scaler.transform(arr_2d)
      predictions = model.predict(arr_2d)

      print(predictions)
      return render_template('index.html', result=str(predictions[0][0]))
        # the result is set, by asking for row=0, column=0. Then cast to string.

  except Exception as e:
    return render_template('index.html', result='error ' + str(e))
if __name__ == '__main__':
    app.run()

