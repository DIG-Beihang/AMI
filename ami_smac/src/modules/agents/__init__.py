REGISTRY = {}

from .agent_selection import SelectionAgent
from .rnn_agent import RNNAgent
from .maa_agent import MAAAgent
from .basic_rnn import BasicRNN
from .basic_linear import BasicLinear
from .policy_oracle import PolicyOracle

REGISTRY["agent_selection"] = SelectionAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["maa"] = MAAAgent
REGISTRY["basic_rnn"] = BasicRNN
REGISTRY["basic_linear"] = BasicLinear
REGISTRY["policy_oracle"] = PolicyOracle