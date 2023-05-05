
from flask import Flask,request,jsonify
import pickle
import numpy as np

model = pickle.load(open('model_rf.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    try:
        return 'Hello World'
    except:
        return jsonify('error:error try again')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        height = request.form.get('height')
        neck = request.form.get('neck')
        waist = request.form.get('waist')

        # result = {'height':height,'neck':neck,'waist':waist}
        # return jsonify(result)

        input_query = np.array([[height,neck,waist]],dtype=float)
        # print(input_query)
        result = model.predict(input_query)[0]

        return jsonify({'Category':str(result)})
    except:
        return jsonify('error:error try again')







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
#     app.run(debug=True)
    app.run(debug=True, port=os.getenv("PORT", default=5000))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
