REGISTRY = {}

from .basic_controller import BasicMAC
from .adv_controller import AdvMAC
from .maa_controller import MAAMAC
from .usenix_controller import UsenixMAC
from .state_controller import STATEMAC
from .state_maa_controller import STATEMAAMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["maa_mac"] = MAAMAC
REGISTRY["adv_mac"] = AdvMAC
REGISTRY["usenix_mac"] = UsenixMAC
REGISTRY["state_mac"] = STATEMAC
REGISTRY["state_maa_mac"] = STATEMAAMAC