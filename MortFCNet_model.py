
import torch
import torch.nn as nn

class MortFCNet(nn.Module):
    def __init__(self, seq_length, input_size,
                 hidden_size_1=256, hidden_size_2=128,
                 hidden_size_3=64, output_size=1):
        super(MortFCNet, self).__init__()

        self.rnn = nn.GRU(input_size, input_size, num_layers = 1, batch_first = True, dropout= 0.2)

        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.ln1 = nn.LayerNorm(hidden_size_1) # LayerNorm after fc1

        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.ln2 = nn.LayerNorm(hidden_size_2) # LayerNorm after fc2

        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.ln3 = nn.LayerNorm(hidden_size_3) # LayerNorm after fc3

        self.fc4 = nn.Linear(hidden_size_3, output_size)

        self.lrelu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        output, _ = self.rnn(x)
        x = output[:, -1, :]

        x = self.fc1(x)
        x = self.ln1(x)  # Layer Normalization
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.ln2(x) # Layer Normalization
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.ln3(x) # Layer Normalization
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.fc4(x)

        return x
