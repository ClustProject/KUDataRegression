import torch
import random
import pickle
import pandas as pd
import numpy as np

import main_regression as mr

# seed 고정
random_seed = 42

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# Case 1. LSTM model (w/o data representation)
config1 = {
        'model': 'LSTM', # Regression에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC', 'DARNN} 중 택 1
        'training': True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
        'best_model_path': './ckpt/lstm.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 144,  # 데이터의 변수 개수, int
            'timestep' : 1, # timestep = window_size
            'shift_size': 1, # shift 정도, int
            'num_classes': 1,  # 분류할 class 개수, int
            'num_layers': 2,  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
            'hidden_size': 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
            'dropout': 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
            'bidirectional': True,  # 모델의 양방향성 여부, bool(default: True)
            'num_epochs': 150,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            'batch_size': 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            'lr': 0.0001,  # learning rate, float(default: 0.001, 범위: 0.1 이하)
            'device': 'cuda',  # 학습 환경, ["cuda", "cpu"] 중 선택
            'need_yhist' : False
        }
}

# Case 2. GRU model (w/o data representation)
config2 = {
        'model': 'GRU', # Regression에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC', 'DARNN} 중 택 1
        'training': True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
        'best_model_path': './ckpt/gru.pt',  # 학습 완료 모델 저장 경로
        'with_representation' : False, # representation 유무, bool (defeault: False)
        'parameter': {
            'input_size': 144,  # 데이터의 변수 개수, int
            'timestep' : 1, # timestep = window_size
            'shift_size': 1, # shift 정도, int
            'num_classes': 1,  # 분류할 class 개수, int
            'num_layers': 2,  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
            'hidden_size': 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
            'dropout': 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
            'bidirectional': True,  # 모델의 양방향성 여부, bool(default: True)
            'num_epochs': 150,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            'batch_size': 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            'lr': 0.0001,  # learning rate, float(default: 0.001, 범위: 0.1 이하)
            'device': 'cuda',  # 학습 환경, ["cuda", "cpu"] 중 선택
            'need_yhist' : False
        }
}

# Case 3. CNN_1D model (w/o data representation)
config3 = {
        'model': 'CNN_1D', # Regression에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC', 'DARNN} 중 택 1
        'training': True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
        'best_model_path': './ckpt/cnn_1d.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 144,  # 데이터의 변수 개수, int
            'timestep' : 1, # timestep = window_size
            'shift_size': 1, # shift 정도, int
            'num_classes': 1,  # 분류할 class 개수, int
            'seq_len': 1,  # 데이터의 시간 길이, int
            'output_channels': 64, # convolution layer의 output channel, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
            'kernel_size': 3, # convolutional layer의 filter 크기, int(default: 3, 범위: 3 이상, 홀수로 설정 권장)
            'stride': 1, # convolution layer의 stride 크기, int(default: 1, 범위: 1 이상)
            'padding': 0, # padding 크기, int(default: 0, 범위: 0 이상)
            'drop_out': 0.1, # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
            'num_epochs': 150,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            'batch_size': 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            'lr': 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
            'device': 'cuda',  # 학습 환경, ["cuda", "cpu"] 중 선택
            'need_yhist' : False
        }
}

# Case 4. DA-RNN model (w/o data representation)
config4 = {
        'model': 'LSTM_FCNs', # Regression에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC', 'DARNN} 중 택 1
        'training': True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
        'best_model_path': './ckpt/lstm_fcn.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 144,  # 데이터의 변수 개수, int
            'timestep' : 1, # timestep = window_size
            'shift_size': 1, # shift 정도, int
            'num_classes': 1,  # 분류할 class 개수, int
            'num_layers': 1,  # recurrent layers의 수, int(default: 1, 범위: 1 이상)
            'lstm_drop_out': 0.4, # LSTM dropout 확률, float(default: 0.4, 범위: 0 이상 1 이하)
            'fc_drop_out': 0.1, # FC dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
            'num_epochs': 150, # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            'batch_size': 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            'lr': 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
            'device': 'cuda',  # 학습 환경, ["cuda", "cpu"] 중 선택
            'need_yhist' : False
        }
}

# Case 5. fully-connected layers (w/ data representation)
config5 = {
        'model': 'FC', # Regression에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC', 'DARNN} 중 택 1
        "training": True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
        "best_model_path": './ckpt/fc.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 144,  # 데이터의 변수 개수(representation 차원), int
            'timestep' : 1, # timestep = window_size
            'shift_size': 1, # shift 정도, int
            'drop_out': 0.1, # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
            'bias': True, # bias 사용 여부, bool(default: True)
            'num_epochs': 150, # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            'batch_size': 32,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            'lr': 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
            'device': 'cuda',  # 학습 환경, ["cuda", "cpu"] 중 선택
            'need_yhist' : False
        }
}

# Case 6. DARNN model (w/o data representation)
config6 = {
        'model': 'DARNN', # Regression에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC', 'DARNN} 중 택 1
        'training': True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
        'best_model_path': './ckpt/darnn.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 144,  # 데이터의 변수 개수, int
            'encoder_hidden_size': 64, # Encoder hidden state의 차원, int(default: 64, 범위: 1 이상)
            'decoder_hidden_size': 64, # Decoder hidden state의 차원, int(default: 64, 범위: 1 이상)
            'timestep': 1, # timestep의 크기, int(default: 16, 범위: 1이상),
            'shift_size' : 1, # Slicing 시 shift 크기
            'encoder_stateful': False, # Encoder의 Stateful 사용여부, bool(default: False)
            'decoder_stateful': False, # Decoder의 Stateful 사용여부, bool(default: False)
            'num_epochs': 150,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            'batch_size': 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            'lr': 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
            'device': 'cuda',  # 학습 환경, ["cuda", "cpu"] 중 선택
            'need_yhist': True
        }
}

import pandas as pd
train_x =pd.read_csv('./data/train_new_energy.csv').values
test_x = pd.read_csv('./data/test_new_energy.csv').values

train_y = pd.read_csv('./data/train_new_energy_y.csv').values
test_y = pd.read_csv('./data/test_new_energy_y.csv').values

train_data = {'x': np.expand_dims(train_x, axis = 1), 'y': train_y}
test_data = {'x': np.expand_dims(test_x, axis = 1), 'y': test_y}

# Case 3. CNN_1D (w/o data representation)
config = config3
data_reg = mr.Regression(config, train_data, test_data, use_representation=True)
model = data_reg.build_model()  # 모델 구축

if config["training"]:
    best_model = data_reg.train_model(model)  # 모델 학습
    data_reg.save_model(best_model, best_model_path=config["best_model_path"])  # 모델 저장

y_true, pred, mse, r2 = data_reg.pred_data(model, best_model_path=config["best_model_path"])  # 예측
print(f'test Loss: {np.round(mse,5)} and R2: {np.round(r2,5)}')
print(f'test RMSE: {np.round(np.sqrt(mse), 4)}')