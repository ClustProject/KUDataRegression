import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models.lstm_fcn import LSTM_FCNs
from models.rnn import RNN_model
from models.cnn_1d import CNN_1D
from models.fc import FC
from models.darnn import DARNN

from models.train_reg_model import Train_Test

import warnings
warnings.filterwarnings('ignore')

# 시계열 데이터 전처리
def sequence_preprocessing(data_x, data_y, timestep, need_yhist, shift_size):

    """
    Window slicing train/test data

    :param data_x: X
    :type config: pandas dataframe

    :param data_y: y
    :type train_data: pandas dataframe

    :param timestep: timestep
    :type test_data: int

    :param need_yhist: need y_hist or not
    :type test_data: boolean

    :param shift_size: Slicing Shift Size
    :type test_data: int
    """

    X = []
    y_hist = []
    targets = []

    for start_idx in range(0, data_x.shape[0] - timestep + 1, shift_size):
        X.append(data_x[start_idx:start_idx + timestep])
        y_hist.append(data_y[start_idx:start_idx + timestep - 1])
        targets.append(data_y[start_idx + timestep - 1])

    X = np.array(X)

    y_hist = np.array(y_hist).reshape(-1, timestep-1, 1)

    targets = np.array(targets)

    if need_yhist == False:
        print(X.shape)
        X = X.transpose(0, 2, 1)

    return X, targets, y_hist


class Regression():
    def __init__(self, config, train_data, test_data, use_representation = False):
        """
        Initialize Classification class and prepare dataloaders for training and testing

        :param config: config
        :type config: dictionary

        :param train_data: train data with X and y
        :type train_data: dictionary

        :param test_data: test data with X and y
        :type test_data: dictionary
        """

        self.config = config
        
        self.model = config['model']
        self.parameter = config['parameter']

        self.train_data = train_data
        self.test_data = test_data

        if use_representation == True:
            self.train_loader, self.valid_loader, self.test_loader = self.get_loaders_repr(train_data=self.train_data,
                                                                                    test_data=self.test_data,
                                                                                    batch_size=self.parameter['batch_size'])
        else:
        # load dataloder
            self.train_loader, self.valid_loader, self.test_loader = self.get_loaders(train_data=self.train_data,
                                                                                        test_data=self.test_data,
                                                                                        batch_size=self.parameter['batch_size'],
                                                                                        timestep=self.parameter['timestep'],
                                                                                        need_yhist=self.parameter['need_yhist'],
                                                                                        shift_size=self.parameter['shift_size'])
            
        # build trainer
        self.trainer = Train_Test(self.config, self.train_data, self.train_loader, self.valid_loader, self.test_loader)


    def build_model(self):
        """
        Build model and return initialized model for selected model_name

        :return: initialized model
        :rtype: model
        """

        # build initialized model
        if self.model == 'LSTM':
            init_model = RNN_model(
                rnn_type='lstm',
                input_size=self.parameter['input_size'],
                hidden_size=self.parameter['hidden_size'],
                num_layers=self.parameter['num_layers'],
                bidirectional=self.parameter['bidirectional'],
                device=self.parameter['device']
            )
        elif self.model == 'GRU':
            init_model = RNN_model(
                rnn_type='gru',
                input_size=self.parameter['input_size'],
                hidden_size=self.parameter['hidden_size'],
                num_layers=self.parameter['num_layers'],
                bidirectional=self.parameter['bidirectional'],
                device=self.parameter['device']
            )
        elif self.model == 'CNN_1D':
            init_model = CNN_1D(
                input_channels=self.parameter['input_size'],
                input_seq=self.parameter['seq_len'],
                output_channels=self.parameter['output_channels'],
                kernel_size=self.parameter['kernel_size'],
                stride=self.parameter['stride'],
                padding=self.parameter['padding'],
                drop_out=self.parameter['drop_out']
            )
        elif self.model == 'LSTM_FCNs':
            init_model = LSTM_FCNs(
                input_size=self.parameter['input_size'],
                num_layers=self.parameter['num_layers'],
                lstm_drop_p=self.parameter['lstm_drop_out'],
                fc_drop_p=self.parameter['fc_drop_out']
            )
        elif self.model == 'FC':
            init_model = FC(
                representation_size=self.parameter['input_size'],
                drop_out=self.parameter['drop_out'],
                bias=self.parameter['bias']
            )
        elif self.model == 'DARNN':
            init_model = DARNN(
                input_size = self.parameter['input_size'],
                encoder_hidden_size = self.parameter['encoder_hidden_size'],
                decoder_hidden_size = self.parameter['decoder_hidden_size'],
                timestep = self.parameter['timestep'],
                stateful_encoder = self.parameter['encoder_stateful'],
                stateful_decoder = self.parameter['decoder_stateful']
            )
        else:
            print('Choose the model correctly')

        return init_model


    def train_model(self, init_model):
        """
        Train model and return best model

        :param init_model: initialized model
        :type init_model: model

        :return: best trained model
        :rtype: model
        """

        print("Start training model")

        # train model
        init_model = init_model.to(self.parameter['device'])

        dataloaders_dict = {'train': self.train_loader, 'val': self.valid_loader}
        criterion = nn.MSELoss()
        optimizer = optim.Adam(init_model.parameters(), lr=self.parameter['lr'])

        best_model = self.trainer.train(init_model, dataloaders_dict, criterion, self.parameter['num_epochs'], optimizer)

        return best_model


    def save_model(self, best_model, best_model_path):
        """
        Save the best trained model

        :param best_model: best trained model
        :type best_model: model

        :param best_model_path: path for saving model
        :type best_model_path: str
        """

        # save model
        torch.save(best_model.state_dict(), best_model_path)


    def pred_data(self, init_model, best_model_path):
        """
        Predict class based on the best trained model
        :param init_model: initialized model
        :type model: model

        :param best_model_path: path for loading the best trained model
        :type best_model_path: str

        :return: predicted value
        :rtype: numpy array

        :return: test mse
        :rtype: float
        """

        print("\nStart testing data\n")

        # load best model
        init_model.load_state_dict(torch.load(best_model_path))

        # get prediction and accuracy
        y_true, pred, mse, r2 = self.trainer.test(init_model, self.test_loader)

        return y_true, pred, mse, r2


    def get_loaders(self, train_data, test_data, batch_size, timestep, need_yhist, shift_size):
        """
        Get train, validation, and test DataLoaders
        
        :param train_data: train data with X and y
        :type train_data: dictionary

        :param test_data: test data with X and y
        :type test_data: dictionary

        :param timestep: timestep
        :type timestep: int

        :param batch_size: batch_size
        :type timestep: int

        :param timestep: need_yhist
        :type timestep: boolean

        :return: train, validation, and test dataloaders
        :rtype: DataLoader
        """

        # 데이터 분할
        x = train_data['x']
        y = train_data['y']
        x_test = test_data['x']
        y_test = test_data['y']

        if need_yhist == True:
            # 데이터 전처리 수행
            x, y, y_hist = sequence_preprocessing(x, y, timestep, need_yhist = need_yhist, shift_size = shift_size)
            x_test, y_test, y_hist_test = sequence_preprocessing(x_test, y_test, timestep, need_yhist = need_yhist, shift_size = shift_size)
            
            # Train, validation Split
            n_train = int(0.8 * len(x))
            x_train, y_train, y_hist_train = x[:n_train], y[:n_train], y_hist[:n_train]
            x_valid, y_valid, y_hist_valid = x[n_train:], y[n_train:], y_hist[n_train:]

            # dataloader 구축
            datasets = []

            for dataset in [(x_train, y_train, y_hist_train), (x_valid, y_valid, y_hist_valid), (x_test, y_test, y_hist_test)]:
                x_data = dataset[0]
                y_data = dataset[1]
                y_hist_data = dataset[2]
                datasets.append(torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data), torch.Tensor(y_hist_data)))

        else:
            # 데이터 전처리 수행
            x, y, _ = sequence_preprocessing(x, y, timestep, need_yhist = need_yhist, shift_size = shift_size)
            x_test, y_test, _ = sequence_preprocessing(x_test, y_test, timestep, need_yhist = need_yhist, shift_size = shift_size)
            
            # Train, validation Split
            n_train = int(0.8 * len(x))
            x_train, y_train = x[:n_train], y[:n_train]
            x_valid, y_valid = x[n_train:], y[n_train:]

            # dataloader 구축
            datasets = []

            for dataset in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
                x_data = dataset[0]
                y_data = dataset[1]
                datasets.append(torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data)))
            
        # dataloader 생성
        trainset, validset, testset = datasets[0], datasets[1], datasets[2]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader


    def get_loaders_repr(self, train_data, test_data, batch_size):
        """
        Get train, validation, and test DataLoaders
        
        :param train_data: train data with X and y
        :type train_data: dictionary

        :param test_data: test data with X and y
        :type test_data: dictionary

        :param timestep: timestep
        :type timestep: int

        :param batch_size: batch_size
        :type timestep: int

        :param timestep: need_yhist
        :type timestep: boolean

        :return: train, validation, and test dataloaders
        :rtype: DataLoader
        """

        # 데이터 분할
        x = train_data['x']
        y = train_data['y']
        x_test = test_data['x']
        y_test = test_data['y']



        # Train, validation Split
        n_train = int(0.8 * len(x))
        x_train, y_train = x[:n_train], y[:n_train]
        x_valid, y_valid = x[n_train:], y[n_train:]

        # dataloader 구축
        datasets = []

        for dataset in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
            x_data = dataset[0]
            y_data = dataset[1]
            datasets.append(torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data)))
            
        # dataloader 생성
        trainset, validset, testset = datasets[0], datasets[1], datasets[2]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader

        