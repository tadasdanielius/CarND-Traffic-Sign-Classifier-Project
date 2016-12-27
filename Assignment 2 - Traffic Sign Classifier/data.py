import pickle
from sklearn.model_selection import train_test_split

TRAIN_SET_DATA_FILE = '../../datasets/traffic-signs-data/train.p'
TEST_SET_DATA_FILE = '../../datasets/traffic-signs-data/test.p'

def load_data(fname):
    with open(fname, mode='rb') as f:
        data = pickle.load(f)
    
    return data['features'], data['labels']

def split_data(x, y, test_size=0.2, random_state=123):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
