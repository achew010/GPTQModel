from collections import OrderedDict
from logging import getLogger

from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear as TritonV2QuantLinear
from ..quantization import FORMAT
from .backend import Backend

backend_dict = OrderedDict({
    Backend.TRITON: TritonV2QuantLinear,
})

format_dict = {
    FORMAT.GPTQ: [Backend.TRITON],
    FORMAT.GPTQ_V2: [Backend.TRITON],
    FORMAT.TRITON: [Backend.TRITON],
}

logger = getLogger(__name__)

# auto select the correct/optimal QuantLinear class
def select_quant_linear(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        backend: Backend,
        format: FORMAT,
        pack: bool = False,
):
    # Handle the case where backend is AUTO.
    if backend == Backend.AUTO:
        allow_backends = format_dict[format]
        for k, v in backend_dict.items():
            in_allow_backends = k in allow_backends
            validate = v.validate(bits, group_size, desc_act, sym, raise_error=False)
            check_pack_func = hasattr(v, "pack") if pack else True
            if in_allow_backends and validate and check_pack_func:
                logger.info(f"Auto choose the fastest one based on quant model compatibility: {v}")
                return v

    # Handle the case where backend is not AUTO.
    if backend == Backend.TRITON:
        logger.info("Using tritonv2 for GPTQ")
        from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
    else:
        raise NotImplementedError("Invalid Backend")

    return QuantLinear
