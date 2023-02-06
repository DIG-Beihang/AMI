import torch.nn as nn
import torch.nn.functional as F


class SelectionAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(SelectionAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_agents)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        # assume stochastic selection policy
        return q