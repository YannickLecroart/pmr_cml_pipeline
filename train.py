import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model():

    x_train_data = np.load("./x_train.npy", allow_pickle=True)
    y_train_data = np.load("./y_train.npy", allow_pickle=True)

    model = RandomForestClassifier(n_estimators = 1000, max_depth=10, verbose=1) #depth 4
    model.fit(x_train_data, y_train_data)
    
    joblib.dump(model, 'model.pkl')


if __name__ == '__main__':
    print('Training PMR model...')
    train_model()