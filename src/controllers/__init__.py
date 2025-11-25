REGISTRY = {}

# normal controllers
from .basic_controller import BasicMAC
from .basic_dc_controller import BasicDCMAC
from .decomposed_controller import DecomposedMAC
from .xtrans_controller import XTransMAC


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_dc_mac"] = BasicDCMAC
REGISTRY["decomposed_mac"] = DecomposedMAC
REGISTRY["xtrans_mac"] = XTransMAC



from .multi_task import XDistralMAC as MultiTaskXDistralMAC

REGISTRY["mt_xdistral_mac"] = MultiTaskXDistralMAC