import torch.nn as nn
import torch.optim as optim

class CharLSTM(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_units, n_layers, output_size):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_units, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_size)
        self.num_layers = n_layers
        self.hidden_units = hidden_units

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = output[:, -1, :] # Take the output of the last time step
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_units),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_units))