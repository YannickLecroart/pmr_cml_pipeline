import joblib
import numpy as np
from xgboost import XGBClassifier

def train_model():

    x_train_data = np.load("./x_train.npy", allow_pickle=True)
    y_train_data = np.load("./y_train.npy", allow_pickle=True)

    model = XGBClassifier(objective='reg:logistic', n_estimators=1000, learning_rate=0.1)
    model.fit(x_train_data, y_train_data, verbose=True)
    
    joblib.dump(model, 'model.pkl')


if __name__ == '__main__':
    print('Training PMR model...')
    train_model()
