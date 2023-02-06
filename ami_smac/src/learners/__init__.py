from .ppo_learner import PPOLearner
from .ppo_usenix_learner import PPOUsenixLearner
from .ppo_icml_learner import PPOIcmlLearner
from .ppo_ami_learner import PPOAMILearner
REGISTRY = {}

REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["ppo_usenix_learner"] = PPOUsenixLearner
REGISTRY["ppo_icml_learner"] = PPOIcmlLearner
REGISTRY["ppo_ami_learner"] = PPOAMILearner