import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.train_model import Train_Test
from models.lstm_fcn import LSTM_FCNs
from models.rnn import RNN_model
from models.cnn_1d import CNN_1D
from models.fc import FC

import warnings
warnings.filterwarnings('ignore')


class Regression():
    def __init__(self, config):
        """
        Initialize Regression class

        :param config: config
        :type config: dictionary
        
        example (training)
            >>> model_name = 'lstm'
            >>> model_params = config.model_config[model_name]
            >>> data_reg = mr.Regression(model_params)
            >>> best_model = data_reg.train_model(train_x, train_y, valid_x, valid_y)  # 모델 학습
            >>> data_reg.save_model(best_model, best_model_path=model_params["best_model_path"])  # 모델 저장
        
        example (testing)

        """

        self.model_name = config['model']
        self.parameter = config['parameter']

        # build trainer
        self.trainer = Train_Test(config)

    def build_model(self):
        """
        Build model and return initialized model for selected model_name

        :return: initialized model
        :rtype: model
        """

        # build initialized model
        if self.model_name == 'LSTM_rg':
            init_model = RNN_model(
                rnn_type='lstm',
                input_size=self.parameter['input_size'],
                hidden_size=self.parameter['hidden_size'],
                num_layers=self.parameter['num_layers'],
                bidirectional=self.parameter['bidirectional'],
                device=self.parameter['device']
            )
        elif self.model_name == 'GRU_rg':
            init_model = RNN_model(
                rnn_type='gru',
                input_size=self.parameter['input_size'],
                hidden_size=self.parameter['hidden_size'],
                num_layers=self.parameter['num_layers'],
                bidirectional=self.parameter['bidirectional'],
                device=self.parameter['device']
            )
        elif self.model_name == 'CNN_1D_rg':
            init_model = CNN_1D(
                input_channels=self.parameter['input_size'],
                input_seq=self.parameter['seq_len'],
                output_channels=self.parameter['output_channels'],
                kernel_size=self.parameter['kernel_size'],
                stride=self.parameter['stride'],
                padding=self.parameter['padding'],
                drop_out=self.parameter['drop_out']
            )
        elif self.model_name == 'LSTM_FCNs_rg':
            init_model = LSTM_FCNs(
                input_size=self.parameter['input_size'],
                num_layers=self.parameter['num_layers'],
                lstm_drop_p=self.parameter['lstm_drop_out'],
                fc_drop_p=self.parameter['fc_drop_out']
            )
        elif self.model_name == 'FC_rg':
            init_model = FC(
                representation_size=self.parameter['input_size'],
                drop_out=self.parameter['drop_out'],
                bias=self.parameter['bias']
            )
        else:
            print('Choose the model correctly')
        return init_model

    def train_model(self, train_x, train_y, valid_x, valid_y):
        """
        Train model and return best model

        :param init_model: initialized model
        :type init_model: model

        :param train_x: input train data 
        :type train_x: numpy array

        :param train_y: target train data 
        :type train_y: numpy array

        :param valid_x: input validation data 
        :type valid_x: numpy array

        :param valid_y: target validation data 
        :type valid_y: numpy array

        :return: best trained model
        :rtype: model
        """

        print(f"Start training model: {self.model_name}")

        # build train/validation dataloaders
        train_loader = self.get_dataloader(train_x, train_y, self.parameter['batch_size'], shuffle=True)
        valid_loader = self.get_dataloader(valid_x, valid_y, self.parameter['batch_size'], shuffle=False)

        # build initialized model
        init_model = self.build_model()
        
        # train model
        dataloaders_dict = {'train': train_loader, 'val': valid_loader}
        best_model = self.trainer.train(init_model, dataloaders_dict)
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

    def pred_data(self, test_x, test_y, y_scaler, best_model_path):
        """
        Predict target value based on the best trained model

        :param test_x: input test data
        :type test_x: numpy array

        :param test_y: target test data
        :type test_y: numpy array

        :param y_scaler: scaler fitted on y variable in train dataset
        :type: MinMaxScaler

        :param best_model_path: path for loading the best trained model
        :type best_model_path: str

        :return: predicted values
        :rtype: numpy array

        :return: test mse
        :rtype: float

        :return: test mae
        :rtype: float
        """

        print(f"Start testing model: {self.model_name}")

        # build test dataloader
        test_loader = self.get_dataloader(test_x, test_y, self.parameter['batch_size'], shuffle=False)

        # build initialized model
        init_model = self.build_model()

        # load best model
        init_model.load_state_dict(torch.load(best_model_path))

        # get predicted values
        pred_data = self.trainer.test(init_model, test_loader)  # shape: (num_of_instance, )

        # inverse normalization to original scale
        true_data = y_scaler.inverse_transform(np.expand_dims(test_y, axis=-1))
        pred_data = y_scaler.inverse_transform(np.expand_dims(pred_data, axis=-1))
        true_data = true_data.squeeze(-1)  # shape=(num_of_instance, )
        pred_data = pred_data.squeeze(-1)  # shape=(num_of_instance, )

        # calculate performance metrics
        mse = mean_squared_error(true_data, pred_data)
        mae = mean_absolute_error(true_data, pred_data)
        
        # merge true value and predicted value
        pred_df = pd.DataFrame()
        pred_df['actual_value'] = true_data
        pred_df['predicted_value'] = pred_data
        return pred_df, mse, mae
    
    def get_dataloader(self, x_data, y_data, batch_size, shuffle):
        """
        Get DataLoader
        
        :param x_data: input data
        :type x_data: numpy array

        :param y_data: target data
        :type y_data: numpy array

        :param batch_size: batch size
        :type batch_size: int

        :param shuffle: shuffle for making batch
        :type shuffle: bool

        :return: dataloader
        :rtype: DataLoader
        """

        # torch dataset 구축
        dataset = torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))

        # DataLoader 구축
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader