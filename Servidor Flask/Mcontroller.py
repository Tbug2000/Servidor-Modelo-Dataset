import pickle
import numpy as np
model = None

def load_model():

    global model
    with open("./ML_Model.pickle",'rb') as f :
        model = pickle.load(f)

def predict_price(estrato,m2,bhk,bath,estado):

    a = [[estrato,m2,bhk,bath,estado]]
    return round(model.predict(a)[0])


# if __name__ == '__main__':
#     load_model()
#     print(predict_price(*[3,120,4,1,1]))