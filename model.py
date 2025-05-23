import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init



class EEGModelMultiAggr1(nn.Module):
    def __init__(self,  
                 in_channels=396,
                 cnn_dim=260,
                 rnn_dim=260, 
                 KS=4, 
                 num_rnn_layers=3,
                 num_rnn_layers_classification=1, # Additional RNN layer for classification
                 num_transformer_layers=8,
                 num_transformer_heads=10,
                 dropout_cnn_pre=0.1,
                 dropout_rnn=0.5, 
                 bidirectional=True, 
                 n_classes=80,
                 num_classes=45, # number of word classes
                 rnn_type="biLSTM",
                 relu1=True,
                 sigmoid1=True):
        
        super(EEGModelMultiAggr1, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=in_channels, 
                              out_channels=cnn_dim, 
                              kernel_size=4, 
                              stride=4,
                              padding=2)
        
        # Activation functions
        self.relu1 = nn.ReLU() if relu1 else None
        self.sigmoid1 = nn.Sigmoid() if sigmoid1 else None
        
        self.predropout = nn.Dropout(dropout_cnn_pre)
        
        self.is_transformer = False
        
        if rnn_type == 'biLSTM':
            self.rnn = nn.LSTM(input_size=cnn_dim, 
                               hidden_size=rnn_dim, 
                               num_layers=num_rnn_layers, 
                               bidirectional=bidirectional,
                               dropout=dropout_rnn)
            
            self.rnn_classification = nn.LSTM(input_size=rnn_dim * 2 if bidirectional else rnn_dim, 
                                              hidden_size=rnn_dim, 
                                              num_layers=num_rnn_layers_classification, 
                                              bidirectional=bidirectional,
                                              dropout=dropout_rnn)
            
        # elif rnn_type == 'biGRU':
        #     self.rnn = nn.GRU(input_size=cnn_dim, 
        #                       hidden_size=rnn_dim, 
        #                       num_layers=num_rnn_layers, 
        #                       bidirectional=bidirectional,
        #                       dropout=dropout_rnn)
            
        #     self.rnn_classification = nn.GRU(input_size=rnn_dim * 2 if bidirectional else rnn_dim, 
        #                                      hidden_size=rnn_dim, 
        #                                      num_layers=num_rnn_layers_classification, 
        #                                      bidirectional=bidirectional,
        #                                      dropout=dropout_rnn)
             
        # elif rnn_type == 'transformer': # bidirectional to false
        #     #print(cnn_dim)
        #     #print(num_transformer_heads)
        #     #print(rnn_dim)
        #     transformer_layer = nn.TransformerEncoderLayer(
        #                         d_model=cnn_dim, 
        #                         nhead=num_transformer_heads, 
        #                         dim_feedforward=rnn_dim, 
        #                         dropout=dropout_rnn)
        #     self.rnn = nn.TransformerEncoder(transformer_layer, 
        #                                      num_layers=num_transformer_layers)
        #     self.is_transformer = True

        mult = 2 if bidirectional else 1

        self.fc_spectrogram = nn.Linear(rnn_dim*mult, n_classes)  # Output for spectrogram reconstruction
        self.fc_classification = nn.Linear(rnn_dim*mult, num_classes)  # Output for word classification

    def forward(self, x):
        #print(x.shape)
        # Input shape: [batch_size, time_step, in_channels]
        x = x.contiguous().permute(0, 2, 1)
        # Shape: [batch_size, in_channels, time_step]

        x = self.conv(x)
        # Shape: [batch_size, cnn_dim, new_time_step]


        if self.relu1:
            x = self.relu1(x)
        if self.sigmoid1:
            x = self.sigmoid1(x)

        x = self.predropout(x)
        # Shape remains: [batch_size, cnn_dim, new_time_step]

        x = x.contiguous().permute(2, 0, 1)
        # Shape: [new_time_step, batch_size, cnn_dim]

        if self.is_transformer:
            x = self.rnn(x)
            # Shape: [new_time_step, batch_size, cnn_dim]
        else:
            x, _ = self.rnn(x)
            # Shape: [new_time_step, batch_size, rnn_dim * 2]  # rnn_dim * 2 for bidirectional

        spectrogram_output = self.fc_spectrogram(x)
        #print(spectrogram_output.shape)
        # Shape: [new_time_step, batch_size, 80]

        spectrogram_output = spectrogram_output.contiguous().permute(1, 0, 2)
        # Shape: [batch_size, new_time_step, 80]

        # Additional RNN layer for classification
        if not self.is_transformer:
            x, _ = self.rnn_classification(x)
            # Shape: [new_time_step, batch_size, rnn_dim * 2]  # rnn_dim * 2 for bidirectional

        # Take the output from the last time step for classification
        last_time_step_output = x[-1, :, :]
        # Shape: [batch_size, rnn_dim * 2]

        classification_output = self.fc_classification(last_time_step_output)
        # Shape: [batch_size, num_classes]
        #print(spectrogram_output.shape)
        #print(classification_output.shape)
        #print('-------')
        return spectrogram_output, classification_output



