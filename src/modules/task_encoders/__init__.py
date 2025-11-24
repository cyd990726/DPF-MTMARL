#### whole task encoder
WHOLE_REGISTRY = {}
from .whole import ForwardModelEncoder
WHOLE_REGISTRY["forward_model"] = ForwardModelEncoder

#### encoder
ENC_REGISTRY = {}
## single task
from .encoders import AttnEncoder
from .encoders import PoolingEncoder
from .encoders import PoolingMPEEncoder
ENC_REGISTRY["attn"] = AttnEncoder
ENC_REGISTRY["pooling"] = PoolingEncoder
ENC_REGISTRY["pooling_mpe"] = PoolingMPEEncoder
## multi task
from .encoders import MultiTaskPoolingEncoder
from .encoders import MultiTaskPoolingMPEEncoder
ENC_REGISTRY["mt_pooling"] = MultiTaskPoolingEncoder
ENC_REGISTRY["mt_pooling_mpe"] = MultiTaskPoolingMPEEncoder

#### decoder
DEC_REGISTRY = {}
from .decoders import MLPDecoder
DEC_REGISTRY["mlp"] = MLPDecoder
