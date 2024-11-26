from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing, is_traceable_wrapper_subclass

from torchao.dtypes.affine_quantized_tensor import register_layout, AffineQuantizedTensor
from torchao.dtypes.uintx.tensor_core_tiled_layout import TensorCoreTiledAQTTensorImpl

from torchao.dtypes.utils import is_device, Layout

aten = torch.ops.aten

def apply_gemlite_quant(weight, group_size=64, bit_width=4, packing_bitwidth=8, contiguous=None, use_hqq=True):
    from torchao.quantization.quant_primitives import ZeroPointDomain, MappingType
    from torchao.dtypes.affine_quantized_tensor import to_affine_quantized_intx
    from torchao.dtypes.uintx.gemlite_layout import GemlitePackedLayout

    if contiguous is None:
        contiguous = True if bit_width < 8 else False

    assert packing_bitwidth in [8, 16, 32], f"gemlite needs packing_bitwidth in [8, 16, 32] but got {packing_bitwidth}"
    assert weight.dtype == torch.float16, f"gemlite only works with dtype torch.float16 but got {weight.dtype}"
    assert group_size in [32, 64, 128, 256, 512, 1024, None]
    assert group_size is None or bit_width != 8, "gemlite only works with group_size=None for bit_width=8"

    out_features, in_features = weight.shape
    group_size = in_features if group_size is None else group_size

    if bit_width != 8:
        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.uint8
        eps = 1e-6
        quant_min = 0
        quant_max = (2**bit_width)-1
        eps = 1e-6
        zero_point_dtype = torch.float16
        zero_point_domain = ZeroPointDomain.FLOAT
    else:
        mapping_type = MappingType.SYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int8
        quant_min = -128
        quant_max = 127
        eps = 1e-5
        zero_point_dtype = None
        zero_point_domain = None
    layout = GemlitePackedLayout(group_size=group_size, bit_width=bit_width, packing_bitwidth=packing_bitwidth, contiguous=contiguous)
    return to_affine_quantized_intx(weight, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, zero_point_dtype=zero_point_dtype, zero_point_domain=zero_point_domain, _layout=layout, use_hqq=use_hqq)

@dataclass(frozen=True)
class GemlitePackedLayout(Layout):
    group_size: Optional[int] = 64
    bit_width: int = 4
    packing_bitwidth: int=8
    contiguous: bool=True


@register_layout(GemlitePackedLayout)
class GemliteAQTTensorImpl(TensorCoreTiledAQTTensorImpl):
    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        gemlite_kwargs: Dict,
        _layout: Layout,
    ):
        kwargs = {}
        kwargs["device"] = packed_weight.device
        kwargs["layout"] = (
            kwargs.get("layout")
            if kwargs.get("layout", False)
            else packed_weight.layout
        )
        kwargs["dtype"] = packed_weight.dtype
        kwargs["requires_grad"] = False
        shape = packed_weight.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        gemlite_kwargs: Dict,
        _layout: Layout,
    ):
        self.packed_weight = packed_weight
        self.scale = scale
        self.zero_point = zero_point
        self.gemlite_kwargs = gemlite_kwargs
        self._layout = _layout
        torch._dynamo.config.inline_inbuilt_nn_modules = False

    def __tensor_flatten__(self):
        return ["packed_weight", "scale", "zero_point"], [self._layout, self.gemlite_kwargs]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight, scale, zero_point = (
            tensor_data_dict["packed_weight"],
            tensor_data_dict["scale"],
            tensor_data_dict["zero_point"],
        )
        _layout, gemlite_kwargs = tensor_attributes
        return cls(packed_weight, scale, zero_point, gemlite_kwargs, _layout)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        from gemlite.core import DType, GemLiteLinearTriton, GEMLITE_ACC_DTYPE, set_autotune

        assert isinstance(_layout, GemlitePackedLayout), f"GemliteAQTTensorImpl only works with GemliteLinearTriton but got {_layout}"
        group_size, bit_width = _layout.group_size, _layout.bit_width

        GEMLITE_ACC_DTYPE[DType.FP16] = DType.FP32
        set_autotune({'GEMV_REVSPLITK':True, 'GEMV':True, 'GEMM_SPLITK':True, 'GEMM':True}, exhaustive=False, use_cuda_graph=False)

        out_features, in_features = int_data.shape
        input_dtype, output_dtype = DType.FP16, DType.FP16
        gemlite_linear = GemLiteLinearTriton(
            bit_width,
            group_size=group_size,
            in_features=in_features,
            out_features=out_features,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        gemlite_linear.pack(int_data, scale, zero_point, bias=None, fma_mode=False, packing_bitwidth=_layout.packing_bitwidth, contiguous=_layout.contiguous)

        list_of_gemlite_kwargs = [
            "unpack_mask",
            "elements_per_sample",
            "input_dtype",
            "output_dtype",
            "acc_dtype",
            "meta_dtype",
            "channel_scale_mode",
            "W_group_mode",
            "data_contiguous",
            "scale_activations",
        ]
        gemlite_kwargs = {}
        for kwarg in list_of_gemlite_kwargs:
            gemlite_kwargs[kwarg] = getattr(gemlite_linear, kwarg) 

        packed_weight, scale, zero_point = gemlite_linear.W_q, gemlite_linear.scales, gemlite_linear.zeros

        return cls(packed_weight, scale, zero_point, gemlite_kwargs, _layout)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        if not is_device("cuda", device):
            raise ValueError(
                f"TensorCoreTiledAQTTensorImpl is only available for cuda device, can't convert to {device}"
            )
        return self.__class__(
            self.packed_weight.to(device),
            self.scale.to(device),
            self.zero_point.to(device),
            self.gemlite_kwargs,
            self._layout,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.packed_weight),
            fn(self.scale),
            fn(self.zero_point),
            self.gemlite_kwargs,
            self._layout,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        if func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        raise NotImplementedError(
            f"GemliteAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO figure out how to do unpacking
        return self.packed_weight, self.scale, self.zero_point

    def get_layout(self) -> Layout:
        return self._layout


def _matmul_type_fn(batch_size: int, bit_width: int, elements_per_sample: int) -> str:
    # print("only gemm")
    # return 'GEMM'
    
    if batch_size > 64:
        return 'GEMM'
    elif batch_size > 1:
        return 'GEMM_SPLITK'
    elif elements_per_sample == 1:
        return 'GEMV_SPLITK'
    elif bit_width<8:
        return 'GEMV_REVSPLITK'
    else:
        return 'GEMV'


def _linear_fp_act_int4_weight_gemlite_impl(input_tensor, weight_tensor, bias):
    from gemlite.core import GEMLITE_TRITON_MAPPING
    input_tensor, scaled_input = weight_tensor.tensor_impl.gemlite_kwargs["scale_activations"](input_tensor)
    batch_size = input_tensor.view(-1, input_tensor.shape[-1]).shape[0]
    out_shape = input_tensor.shape[:-1] + (weight_tensor.shape[0],)
    matmul_type = _matmul_type_fn(batch_size, weight_tensor._layout.bit_width, weight_tensor.tensor_impl.gemlite_kwargs["elements_per_sample"])

    out = (
        GEMLITE_TRITON_MAPPING[matmul_type]
        .forward(
            input_tensor.view(-1, input_tensor.shape[-1]),
            weight_tensor.tensor_impl.packed_weight,
            weight_tensor.tensor_impl.scale,
            weight_tensor.tensor_impl.zero_point,
            scaled_input,
            weight_tensor._layout.bit_width,
            weight_tensor._layout.group_size,
            weight_tensor.tensor_impl.gemlite_kwargs["unpack_mask"],
            weight_tensor.tensor_impl.gemlite_kwargs["elements_per_sample"],
            weight_tensor.tensor_impl.gemlite_kwargs["input_dtype"].value,
            weight_tensor.tensor_impl.gemlite_kwargs["output_dtype"].value,
            weight_tensor.tensor_impl.gemlite_kwargs["acc_dtype"].value,
            weight_tensor.tensor_impl.gemlite_kwargs["meta_dtype"].value,
            weight_tensor.tensor_impl.gemlite_kwargs["channel_scale_mode"],
            weight_tensor.tensor_impl.gemlite_kwargs["W_group_mode"],
            weight_tensor.tensor_impl.gemlite_kwargs["data_contiguous"],
        )
    )
    

    return out.view(out_shape)


def _linear_fp_act_int4_weight_gemlite_check(
    input_tensor, weight_tensor, bias
):
    return (
    # input is native fp16 tensor
    not is_traceable_wrapper_subclass(input_tensor)
    and input_tensor.dtype == torch.float16
    # weight is gemlite layout
    and isinstance(weight_tensor, AffineQuantizedTensor)
    and isinstance(weight_tensor._layout, GemlitePackedLayout)
    )
