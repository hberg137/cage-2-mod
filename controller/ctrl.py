import torch
from torch.utils.data import Dataset

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset object that will contain our sliding window data
class LSTMDataset(Dataset):
    def __init__(self, states, labels):
        self.states = states
        self.labels = labels

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        state = torch.tensor(self.states[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return state, label

class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
        # Output Activation
        self.out_act = torch.nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.out_act(self.fc(out[:, -1, :]))
        # out.size() --> 100, 10
        return out
