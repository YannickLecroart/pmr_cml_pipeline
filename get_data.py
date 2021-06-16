import pickle


def get_data(self):
    
    f = open("pmr_data.pkl", "wb")
    pickle.dump(pmr_data, f)
    f.close()
    
     
if __name__ == '__main__':

    print('Getting PMR data...')
    get_data()
    

