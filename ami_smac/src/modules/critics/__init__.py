from .centralV import CentralVCritic
from .central_oracle import CentralVOracle

REGISTRY = {}


REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["cv_oracle"] = CentralVOracle


