"""Feed-forward neural network for three-class sentiment classification."""

import torch.nn as nn  
import torch.nn.functional as F  


class ANN(nn.Module):
    """Two hidden layers, each followed by batch normalisation, ReLU and dropout."""

    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=256, output_dim=3, dropout_p=0.5):
        """Declare the layers. input_dim differs per representation: 5000 for TF-IDF, 100 for Word2Vec."""
        super().__init__()  

        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # Weighted sums from the features to the first hidden layer
        self.bn1 = nn.BatchNorm1d(hidden_dim1)  # Keeps the scale of those sums stable across epochs
        self.dropout1 = nn.Dropout(p=dropout_p)  # Randomly silences 30% of the neurons while training

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(p=dropout_p)

        self.fc3 = nn.Linear(hidden_dim2, output_dim)  # One output per sentiment class

    def forward(self, x):
        """Send one batch through the network and return the raw class scores."""
        x = self.fc1(x)  # Weighted sums into the first hidden layer
        x = self.bn1(x)  # Normalise before the activation
        x = F.relu(x)  
        x = self.dropout1(x)  # Dropout to silence some neurons while training

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x) 
        return x
