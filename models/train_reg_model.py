import time
import copy

import torch
import torch.nn as nn
from sklearn.metrics import r2_score

class Train_Test():
    def __init__(self, config, train_data, train_loader, valid_loader, test_loader): ##### config는 jupyter 파일을 참고
        """
        Initialize Train_Test class

        :param config: configuration
        :type config: dictionary

        :param train_data: train_data (x, y)
        :type config: dictionary

        :param train_loader: train dataloader
        :type config: DataLoader

        :param valid_loader: validation dataloader
        :type config: DataLoader

        :param test_loader: test dataloader
        :type config: DataLoader
        """

        self.config = config
        self.train_data = train_data

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.model = self.config['model']
        self.parameter = self.config['parameter']
        self.input_size = self.parameter['input_size']


    def train(self, model, dataloaders, criterion, num_epochs, optimizer):
        """
        Train the model

        :param model: initialized model
        :type model: model

        :param dataloaders: train & validation dataloaders
        :type dataloaders: dictionary

        :param criterion: loss function for training
        :type criterion: criterion

        :param num_epochs: the number of train epochs
        :type num_epochs: int

        :param optimizer: optimizer used in training
        :type optimizer: optimizer

        :return: trained model
        :rtype: model
        """

        since = time.time() 

        val_loss_history = []

        best_loss = 999999999
        best_model_wts = copy.deepcopy(model.state_dict()) ##### 모델의 초기 Weight값 (각 Layer 별 초기 Weight값이 저장되어 있음)

        for epoch in range(num_epochs):
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))

            # 각 epoch마다 순서대로 training과 validation을 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  
                else:
                    model.eval()   

                running_loss = 0.0
                running_total = 0

                if self.parameter['need_yhist'] == True:
                    # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                    for inputs, targets, y_hist in dataloaders[phase]:
                        inputs = inputs.to(self.parameter['device'])
                        targets = targets.to(self.parameter['device'])
                        y_hist = y_hist.to(self.parameter['device'])
                        
                        # parameter gradients를 0으로 설정
                        optimizer.zero_grad()

                        # training 단계에서만 gradient 업데이트 수행
                        with torch.set_grad_enabled(phase == 'train'):

                            # input을 model에 넣어 output을 도출한 역정규화 후, loss를 계산함
                            outputs = model(inputs, y_hist)
                            outputs = outputs.squeeze(1)
                            loss = criterion(outputs, targets)

                            # backward (optimize): training 단계에서만 수행
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # batch내 loss를 축적함
                        running_loss += loss.item() * inputs.size(0)
                        running_total += targets.size(0)

                else:
                    # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                    for inputs, targets in dataloaders[phase]:
                        inputs = inputs.to(self.parameter['device'])
                        targets = targets.to(self.parameter['device'])
                        
                        # parameter gradients를 0으로 설정
                        optimizer.zero_grad()

                        # training 단계에서만 gradient 업데이트 수행
                        with torch.set_grad_enabled(phase == 'train'):

                            # input을 model에 넣어 output을 도출한 역정규화 후, loss를 계산함
                            outputs = model(inputs)
                            outputs = outputs.squeeze(1)
                            loss = criterion(outputs, targets)

                            # backward (optimize): training 단계에서만 수행
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # batch내 loss를 축적함
                        running_loss += loss.item() * inputs.size(0)
                        running_total += targets.size(0)

                # epoch의 loss 및 accuracy 도출
                epoch_loss = running_loss / running_total

                # log 출력
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                if phase == 'val':
                    val_loss_history.append(epoch_loss)

        # 전체 학습 시간 계산 (학습이 완료된 후)
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

        # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
        model.load_state_dict(best_model_wts)

        return model


    def test(self, model, test_loader):
        """
        Predict classes for test dataset based on the trained model

        :param model: best trained model
        :type model: model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: predicted values
        :rtype: numpy array

        :return: MSE
        :rtype: float
        """

        model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)

        with torch.no_grad():

            preds = []
            y_true = []
            criterion = nn.MSELoss()

            if self.parameter['need_yhist'] == True:
                for inputs, targets, y_hist in test_loader:
                    inputs = inputs.to(self.parameter['device'])
                    targets = targets.to(self.parameter['device'])
                    y_hist = y_hist.to(self.parameter['device'])

                    pred = model(inputs, y_hist)

                    preds.extend(pred.detach().cpu().numpy())
                    y_true.extend(targets.detach().cpu().numpy())

            else:
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.parameter['device'])
                    targets = targets.to(self.parameter['device'])

                    pred = model(inputs)

                    preds.extend(pred.detach().cpu().numpy())
                    y_true.extend(targets.detach().cpu().numpy())

            preds = torch.tensor(preds).reshape(-1)
            y_true = torch.tensor(y_true)
            
            mse = criterion(preds, y_true).item()
            r2 = r2_score(preds.numpy(), y_true.numpy())
        return y_true, preds, mse, r2