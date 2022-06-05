import torch
import torch.nn.functional as F

from torch import nn

# InputAttention Mechanism (Encoder)
class InputAttentionEncoder(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, timestep, stateful=False):
        """
        Initialize TemporalAttentionDecoder Class
        
        :param input_size: number of features
        :type config: int

        :param encoder_hidden_size: number of LSTM units
        :type config: int

        :param timestep: number of timesteps
        :type config: int
            
        :param stateful: decides whether to initialize cell state of new time window with values of the last cell state
                         of previous time window or to initialize it with zeros
        :type config: Boolean    

        return encoded input
        rtype: (batch_size, timestep, encoder_hidden_size) tensor
        """

        super(self.__class__, self).__init__()
        self.input_size = input_size # Input 변수의 개수: 81
        self.encoder_hidden_size = encoder_hidden_size # Encoder hidden state의 개수: 64
        self.timestep = timestep # Timestep의 크기: 16
        
        self.encoder_lstm = nn.LSTMCell(input_size=self.input_size, hidden_size=self.encoder_hidden_size) # input : 81 , hidden : 64
        
        #equation 8 matrices
        self.W_e = nn.Linear(2 * self.encoder_hidden_size, self.timestep) # 2 *64 -> 16 (16 x 128의 형태)
        self.U_e = nn.Linear(self.timestep, self.timestep, bias=False) # 16 -> 16
        self.v_e = nn.Linear(self.timestep, 1, bias=False) # 16 -> 1
    

    def forward(self, inputs): # inputs : batch_x (128, 16, 81)
        encoded_inputs = torch.zeros((inputs.size(0), self.timestep, self.encoder_hidden_size)).cuda() ### out: (128, 16, 64) <- Encoder의 Input 초기화
        
        #initiale hidden states
        h_tm1 = torch.zeros((inputs.size(0), self.encoder_hidden_size)).cuda() # out: (128, 64) <- LSTM hidden state
        s_tm1 = torch.zeros((inputs.size(0), self.encoder_hidden_size)).cuda() # out: (128, 64) <- LSTM cell state
        
        for t in range(self.timestep):
            #concatenate hidden states
            h_c_concat = torch.cat((h_tm1, s_tm1), dim=1) # out: (128, 128)
            
            #attention weights for each k in N (equation 8)
            x = self.W_e(h_c_concat).unsqueeze_(1).repeat(1, self.input_size, 1) # out: (128, 81, 16) // unsqueeze 결과 (128, 16) --> (128, 1, 16) // repeat 결과 (128, 1, 16) --> (128, 81, 16)
            y = self.U_e(inputs.permute(0, 2, 1)) # out: (128, 81, 16) <- (128, 81, 16) * (16, 16)
            z = torch.tanh(x + y) # out: (128, 81, 16)
            e_k_t = torch.squeeze(self.v_e(z)) # out: (128, 81) <- (128, 81, 16) * (16, 1) - Squeeze - (128, 81)
        
            #normalize attention weights (equation 9)
            alpha_k_t = F.softmax(e_k_t, dim=1) # out: (128, 81)
            
            #weight inputs (equation 10)
            weighted_inputs = alpha_k_t * inputs[:, t, :] ### t = 0 일단 가정 (t는 16까지 값으로 timestep을 의미) --> out : (128, 81)
    
            #calculate next hidden states (equation 11)
            h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1)) # out: (128, 64), (128, 64) // nn.LSTMCell은 Input of shape batch × input dimension;, A tuple of LSTM hidden states of shape batch x hidden dimensions.
            
            encoded_inputs[:, t, :] = h_tm1 # encoded inputs가 (128, 16, 64)인데, (128, t, 64) 자리에 (128, 64)를 삽입
            
        return encoded_inputs  ### out: (128, 16, 64) --> 이는 결국 h_1~h_T의 concat된 것을 return (Temporal Attention의 Input 부분)


# TemporalAttention Mechanism Decoder
class TemporalAttentionDecoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, timestep, stateful=False):
        """
        Initialize TemporalAttentionDecoder Class
        :param encoder_hidden_size: number of encoder LSTM units
        type config: int

        :param decoder_hidden_size: number of decoder LSTM units
        type config: int    

        :param timestep: number of timesteps
        type config: int

        :param stateful: decides whether to initialize cell state of new time window with values of the last cell state
                         of previous time window or to initialize it with zeros
        type config: Boolean

        return encoded input
        rtype: (batch_size, 1) tensor
        """

        super(self.__class__, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size # Encoder LSTM Unit의 수: 64
        self.decoder_hidden_size = decoder_hidden_size # Decoder LSTM Unit의 수: 64
        self.timestep = timestep # Timestep의 크기: 16
        self.stateful = stateful # T/F: F
        
        self.decoder_lstm = nn.LSTMCell(input_size=1, hidden_size=self.decoder_hidden_size) # Input_size : 1 // Hidden_size : 64
        
        #equation 12 matrices
        self.W_d = nn.Linear(2 * self.decoder_hidden_size, self.encoder_hidden_size) # 128 --> 64
        self.U_d = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False) # 64 --> 64
        self.v_d = nn.Linear(self.encoder_hidden_size, 1, bias = False) # 64 --> 1
        
        #equation 15 matrix
        self.w_tilda = nn.Linear(self.encoder_hidden_size + 1, 1) # 65 --> 1
        
        #equation 22 matrices
        self.W_y = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.decoder_hidden_size) # 128 --> 64
        self.v_y = nn.Linear(self.decoder_hidden_size, 1) # 64 --> 1
        

    def forward(self, encoded_inputs, y): # encoded_inputs : (128, 16, 64) // y(batch_y_h) : (128, 16, 1)
        
        #initializing hidden states
        d_tm1 = torch.zeros((encoded_inputs.size(0), self.decoder_hidden_size)).cuda() # out : (128, 64) - decoder hidden state
        s_prime_tm1 = torch.zeros((encoded_inputs.size(0), self.decoder_hidden_size)).cuda() # out : (128, 64) - cell state
        
        for t in range(self.timestep-1):
            
            #concatenate hidden states
            d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1) # out : (128, 128)
            
            #temporal attention weights (equation 12)
            x1 = self.W_d(d_s_prime_concat).unsqueeze_(1).repeat(1, encoded_inputs.shape[1], 1) # out : (128, 16, 64)
            y1 = self.U_d(encoded_inputs) # out : (128, 16, 64)
            z1 = torch.tanh(x1 + y1) # out : (128, 16, 64)
            l_i_t = self.v_d(z1) # out : (128, 16, 1) - 1개 변수의 l_i값을 의미
            
            #normalized attention weights (equation 13)
            beta_i_t = F.softmax(l_i_t, dim=1) # out : (128, 16, 1)
            
            #create context vector (equation_14)
            c_t = torch.sum(beta_i_t * encoded_inputs, dim=1) # out : (128, 64)  
            
            #concatenate c_t and y_t
            y_c_concat = torch.cat((c_t, y[:, t, :]), dim=1) # out : (128, 65) // Y는 y_batch_h를 의미 (128, 16, 1)
            
            #create y_tilda
            y_tilda_t = self.w_tilda(y_c_concat) # out : (128, 65)
            
            #calculate next hidden states (equation 16)
            d_tm1, s_prime_tm1 = self.decoder_lstm(y_tilda_t, (d_tm1, s_prime_tm1)) # out : (128, 64) // s_prime_tm1 : (128, 64)
        
        #concatenate context vector at step T and hidden state at step T
        d_c_concat = torch.cat((d_tm1, c_t), dim=1) # out : (128, 128)

        #calculate output
        y_Tp1 = self.v_y(self.W_y(d_c_concat)) # out : (128, 1)
        return y_Tp1


# Overall DARNN
class DARNN(nn.Module):
    """
    Initialize DARNN Class

    :param input_size: number of time serieses
    :type config: int    

    :param encoder_hidden_size: number of LSTM units
    :type config: int

    :param decoder_hidden_size: number of deocder LSTM units
    :type config: int

    :param timestep: number of timesteps
    :type config: int

    :param stateful_encoder: decides whether to initialize cell state of new time window with values of the last cell state
                             of previous time window or to initialize it with zeros
    :type config: Boolean    

    :param stateful_decoder: decides whether to initialize cell state of new time window with values of the last cell state
                             of previous time window or to initialize it with zeros
    :type config: Boolean    

    return predicted value
    rtype: (batch_size, 1) tensor
    """

    def __init__(self, input_size, encoder_hidden_size, decoder_hidden_size, timestep, stateful_encoder=False, stateful_decoder=False):
        
        super(self.__class__, self).__init__()
        self.encoder = InputAttentionEncoder(input_size, encoder_hidden_size, timestep, stateful_encoder).cuda()
        self.decoder = TemporalAttentionDecoder(encoder_hidden_size, decoder_hidden_size, timestep, stateful_decoder).cuda()
        

    def forward(self, X_history, y_history): # X_history : batch_x // y_history : batch_y_h
        
        out = self.decoder(self.encoder(X_history), y_history) # (128, 1)
        return out