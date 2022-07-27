##########################
# Default List Information
modelTestconfig={
    "LSTM_rg":{# Case 1. LSTM model (w/o data representation)
        'num_layers': 2,  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
        'hidden_size': 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상) (+Modified)
        'dropout': 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        'bidirectional': True,  # 모델의 양방향성 여부, bool(default: True)   
        "lr":0.0001
    },
    "GRU_rg":{# Case 2. GRU model (w/o data representation)
        'num_layers': 2,  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
        'hidden_size': 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
        'dropout': 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        'bidirectional': True,  # 모델의 양방향성 여부, bool(default: True)
        "lr":0.0001
        
    },
    "CNN_1D_rg":{# Case 3. CNN_1D model (w/o data representation)
        'output_channels': 64, # convolution layer의 output channel, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
        'kernel_size': 3, # convolutional layer의 filter 크기, int(default: 3, 범위: 3 이상, 홀수로 설정 권장)
        'stride': 1, # convolution layer의 stride 크기, int(default: 1, 범위: 1 이상)
        'padding': 0, # padding 크기, int(default: 0, 범위: 0 이상)
        'drop_out': 0.1, # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        "lr":0.0001
    },
    "LSTM_FCNs_rg":{#Case 4. LSTM_FCNs model (w/o data representation)
        'num_layers': 2,  # # recurrent layers의 수, int(default: 2, 범위: 1 이상) (+Modified)
        'lstm_drop_out': 0.4, # LSTM dropout 확률, float(default: 0.4, 범위: 0 이상 1 이하)
        'fc_drop_out': 0.1, # FC dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        "lr":0.0001
    },
    "FC_rg":{# Case 5. fully-connected layers (w/ data representation)
        'drop_out': 0.1, # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        'bias': True,# bias 사용 여부, bool(default: True)
        "lr":0.0001}  
}

import pickle
import pandas as pd
def getTrainDataFromFilesForRegression(folderAddress, model_name):
    if model_name in ["LSTM_rg","GRU_rg", "CNN_1D_rg","LSTM_FCNs_rg"] :
        # raw time series data
        train_x = pickle.load(open(folderAddress+'x_train.pkl', 'rb'))
        train_y = pickle.load(open(folderAddress+'y_train.pkl', 'rb'))
        test_x = pickle.load(open(folderAddress+'x_test.pkl', 'rb'))
        test_y = pickle.load(open(folderAddress+'y_test.pkl', 'rb'))

        print(train_x.shape) 
        print(train_y.shape) 
        print(test_x.shape)  
        print(test_y.shape)  
        print("inputSize(train_x.shape[1]):", train_x.shape[1]) # input size
        print("sequenceLenth (train_x.shape[2]):", train_x.shape[2] )# seq_length
    
    if model_name in["FC_rg"]:
        # representation data
        train_x = pd.read_csv(folderAddress+'ts2vec_repr_train.csv')
        train_y = pickle.load(open(folderAddress+'y_train.pkl', 'rb'))

        test_x = pd.read_csv(folderAddress+'ts2vec_repr_test.csv')
        test_y = pickle.load(open(folderAddress+'y_test.pkl', 'rb'))

    return train_x, train_y,test_x, test_y