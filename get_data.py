import pickle
import joblib


def get_data():

    pmr_data = joblib.load("./pmr_data.pkl")
    
    f = open("pmr_data.pkl", "wb")
    pickle.dump(pmr_data, f)
    f.close()
    
     
if __name__ == '__main__':

    print('Getting PMR data...')
    get_data()
    

