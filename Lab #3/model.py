from torch import nn

class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, scale_grad_by_freq=True)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        return out, hidden


def initMyModel(vocab_size):
    embedding_size = 256
    hidden_size = 128
    num_layers = 3
    return MyModel(vocab_size, embedding_size, hidden_size, num_layers)