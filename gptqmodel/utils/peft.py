import warnings
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import torch
from peft import PeftConfig, PeftModel, PeftType, get_peft_model
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.tuners.lora import LoraConfig, LoraLayer, LoraModel

from ..models.base import BaseGPTQModel
# from ..nn_modules.qlinear.qlinear_cuda import QuantLinear as QuantLinearCuda
# from ..nn_modules.qlinear.qlinear_cuda_old import QuantLinear as QuantLinearCudaOld
# from ..nn_modules.qlinear.qlinear_exllama import QuantLinear as QuantLinearExllama
# from ..nn_modules.qlinear.qlinear_exllama import QuantLinear as QuantLinearExllamaV2
from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear as QuantLinearTriton


LinearLayer = Union[
    torch.nn.Linear,
    # QuantLinearCuda,
    # QuantLinearCudaOld,
    # QuantLinearExllama,
    # QuantLinearExllamaV2,
    QuantLinearTriton,
]


class GPTQLoraConfig(LoraConfig):
    injected_fused_attention: bool = False
    injected_fused_mlp: bool = False


def _get_linear_feature_count(linear_layer: LinearLayer) -> Tuple[int, int]:
    in_features = getattr(linear_layer, "in_features", getattr(linear_layer, "infeatures"))
    out_features = getattr(linear_layer, "out_features", getattr(linear_layer, "outfeatures"))
    return in_features, out_features


def _get_weight(linear_layer: LinearLayer) -> torch.Tensor:
    return getattr(linear_layer, "weight", getattr(linear_layer, "qweight"))


class GPTQLoraLinear(torch.nn.Linear, LoraLayer):
    def __init__(
        self,
        adapter_name: str,
        linear_module: LinearLayer,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        in_features, out_features = _get_linear_feature_count(linear_module)
        torch.nn.Linear.__init__(self, in_features, out_features)
        LoraLayer.__init__(self, in_features, out_features)

        self.linear_module = linear_module

        delattr(self, "weight")
        self.weight = _get_weight(linear_module)
        delattr(self, "bias")

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            assert hasattr(linear_module, "weight")
            linear_module.weight.data = linear_module.weight.data.T

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            torch.nn.init.xavier_uniform_(self.lora_A[adapter_name].weight)
            torch.nn.init.zeros_(self.lora_B[adapter_name].weight)

    def merge(self):
        raise NotImplementedError("gptq model not support merge lora adapter")

    def unmerge(self):
        raise NotImplementedError("gptq model not support unmerge lora adapter")

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return self.linear_module(x)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = self.linear_module(x)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = self.linear_module(x)

            lora_B = self.lora_B[self.active_adapter]
            lora_A = self.lora_A[self.active_adapter]
            lora_dropout = self.lora_dropout[self.active_adapter]
            scale = self.scaling[self.active_adapter]

            x = x.type_as(lora_A.weight.data)
            adapter_result = (lora_B(lora_A(lora_dropout(x))) * scale).type_as(result)
            result += adapter_result
        else:
            result = self.linear_module(x)

        result = result.to(previous_dtype)

        return result


class GPTQLoraModel(LoraModel):
    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        if not isinstance(new_module, GPTQLoraLinear):
            new_module.weight = old_module.weight
            if hasattr(old_module, "bias"):
                if old_module.bias is not None:
                    new_module.bias = old_module.bias

            if getattr(old_module, "state", None) is not None:
                new_module.state = old_module.state
                new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                device = (list(old_module.parameters()) + list(old_module.buffers()))[0].device
                module.to(device)


    def merge_adapter(self):
        raise NotImplementedError("gptq model not support merge ada lora adapter")

    def unmerge_adapter(self):
        raise NotImplementedError("gptq model not support unmerge ada lora adapter")

    def merge_and_unload(self):
        raise NotImplementedError("gptq model not support merge and unload")


def find_all_linear_names(
    model: BaseGPTQModel,
    ignore: Optional[List[str]] = None,
    ignore_lm_head: bool = True,
):
    if not ignore:
        ignore = []
    lm_head_name = model.lm_head_name
    if ignore_lm_head and lm_head_name not in ignore:
        ignore.append(lm_head_name)
    results = set()
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            res = n.split(".")[-1]
            if res not in ignore:
                results.add(res)
    return list(results)


@contextmanager
def hijack_peft_mappings():
    PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.LORA] = GPTQLoraConfig
    PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GPTQLoraModel

    try:
        yield
    except:
        PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.LORA] = GPTQLoraConfig
        PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GPTQLoraModel
        raise
    finally:
        PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.LORA] = GPTQLoraConfig
        PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = GPTQLoraModel

def get_gptq_peft_model(
    model: BaseGPTQModel,
    peft_config: PeftConfig = None,
    model_id: str = None,
    adapter_name: str = "default",
    auto_find_all_linears: bool = True,
    train_mode: bool = False,
):
    if train_mode and not model.trainable:
        model.enable_trainable_mode()
    if train_mode and not peft_config:
        raise ValueError("peft_config not specified when in train mode.")
    if not train_mode and not model_id:
        raise ValueError("model_id(where to load adapters) not specified when in inference mode.")

    if train_mode:
        peft_type = peft_config.peft_type
        if not isinstance(peft_type, str):
            peft_type = peft_type.value
        if peft_type in [PeftType.LORA.value]:
            if auto_find_all_linears:
                peft_config.target_modules = find_all_linear_names(model, ignore_lm_head=True)
            if peft_type == PeftType.LORA.value and not isinstance(peft_config, GPTQLoraConfig):
                peft_config = GPTQLoraConfig(**peft_config.to_dict())

    with hijack_peft_mappings():
        try:
            if train_mode:
                peft_model = get_peft_model(model.model, peft_config, adapter_name=adapter_name)
            else:
                peft_model = PeftModel.from_pretrained(model.model, model_id, adapter_name)
        except:
            raise NotImplementedError(
                f"{model.__class__.__name__} not support {peft_config.peft_type.value} peft type yet."
            )

    return peft_model


__all__ = [
    "GPTQLoraConfig",
    "GPTQLoraModel",
    "find_all_linear_names",
    "get_gptq_peft_model",
]