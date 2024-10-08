# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
# ********************************************************
from enum import Enum

# CUDA extension
import numpy as np
import torch

# Triton
import triton.language as tl
from triton.testing import do_bench
from .triton_kernels import *


class DType(Enum):
    FP16 = "FP16"
    BF16 = "BF16"
    FP32 = "FP32"
    INT8 = "INT8"
    INT32 = "INT32"
    FP16D8 = "FP16D8i"  # dynamic quantization

###################################################################################################################################
# CUDA backend
###################################################################################################################################


###################################################################################################################################
# Triton backend
###################################################################################################################################
def eval_time(fct, params, warmup=25, rep=200, fast_flush=True, return_mode="min"):
    if isinstance(params, dict):
        return do_bench(
            lambda: fct(**params),
            warmup=warmup,
            rep=rep,
            fast_flush=fast_flush,
            return_mode=return_mode,
        )
    if isinstance(params, list):
        return do_bench(
            lambda: fct(*params),
            warmup=warmup,
            rep=rep,
            fast_flush=fast_flush,
            return_mode=return_mode,
        )


GEMLITE_TRITON_CACHE = {}

GEMLITE_TRITON_MAPPING = {
    ("FP16", "GEMV"): gemv_A16fWnO16f_int32packing,
    ("FP16", "GEMM"): gemm_A16fWnO16f_int32packing,
    ("BF16", "GEMM"): gemm_A16fWnO16f_int32packing,
}


# Triton
class GemLiteLinearTriton(torch.nn.Module):
    def __init__(
        self,
        W_nbits,
        group_size,
        in_features,
        out_features,
        input_dtype=DType.FP16,
        output_dtype=DType.FP16,
        acc_dtype=None,
    ):
        self._SUPPORTED_BITS_TRITON = [1, 2, 4, 8]

        super().__init__()
        if W_nbits not in self._SUPPORTED_BITS_TRITON:
            raise NotImplementedError("Only 2,4,8 W_nbits are supported.")
        if in_features % 128 != 0 or out_features % 128 != 0:
            raise NotImplementedError("Invalid input shapes")

        self.in_features = in_features
        self.out_features = out_features
        self.orig_shape = (out_features, in_features)
        self.W_nbits = W_nbits
        self.group_size = group_size if group_size != -1 else in_features
        self.unpack_mask = 2**self.W_nbits - 1
        self.elements_per_sample = 32 // self.W_nbits
        self.signature = (in_features, out_features, W_nbits, group_size)

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

        self.compute_dtype = None
        if input_dtype == DType.FP16 and output_dtype == DType.FP16:
            self.kernels = [gemm_A16fWnO16f_int32packing, gemv_A16fWnO16f_int32packing]
            self.compute_dtype = torch.float16
        if input_dtype == DType.BF16 and output_dtype == DType.BF16:
            self.kernels = [gemm_A16fWnO16f_int32packing]
            self.compute_dtype = torch.bfloat16

        if self.compute_dtype is None:
            raise NotImplementedError(
                "Unsupport settings: ",
                (self.input_dtype, self.output_dtype, self.W_nbits),
            )

        if acc_dtype is None:
            acc_dtype = (
                DType.FP16 if (self.compute_dtype == torch.float16) else DType.FP32
            )
        self.acc_dtype = torch.float16 if (acc_dtype == DType.FP16) else torch.float32

        self.dtype = self.output_dtype

        with torch.device("meta"):
            self.register_buffer(
                "W_q",
                torch.zeros(
                    (self.in_features // 32 * self.W_nbits, self.out_features),
                    dtype=torch.int32,
                ),
            )
            self.register_buffer(
                "scales",
                torch.zeros(
                    int(np.ceil(self.in_features / self.group_size)),
                    self.out_features,
                    dtype=self.compute_dtype,
                ),
            )
            self.register_buffer(
                "zeros",
                torch.zeros(
                    int(np.ceil(self.in_features / self.group_size)),
                    self.out_features,
                    dtype=self.compute_dtype,
                ),
            )

        self.forward = self.forward_auto

    # Pack data: following the same logic as: https://github.com/LeiWang1999/AutoGPTQ.tvm/blob/dcd135b9784b9f98235fc91467fe3c3c8afa34fc/auto_gptq/nn_modules/qlinear_triton.py#L413-L419
    def pack(self, W_q, scales, zeros, bias=None):
        W_q = W_q.reshape(self.orig_shape).t().contiguous().to(torch.int32)
        self.W_q = torch.zeros(
            (W_q.shape[0] // 32 * self.W_nbits, W_q.shape[1]),
            dtype=torch.int32,
            device=W_q.device,
        )

        step = 32 // self.W_nbits
        i, row = 0, 0
        while row < self.W_q.shape[0]:
            shift = 0
            for j in range(i, i + step):
                self.W_q[row] |= W_q[j] << shift
                shift += self.W_nbits
            i += step
            row += 1

        self.W_q = self.W_q.contiguous()
        self.scales = scales.reshape((self.out_features, -1)).t().contiguous()
        self.zeros = zeros.reshape((self.out_features, -1)).t().contiguous()
        self.bias = (
            None
            if (bias is None)
            else torch.nn.Parameter(
                bias.to(device=self.W_q.device, dtype=self.compute_dtype)
            )
        )
        self.device = self.W_q.device
        return self

    # Warm up all the selected kernels
    def warmup(self, signature, args):
        global GEMLITE_TRITON_CACHE
        t = []
        for _kernel in self.kernels:
            if signature[0] > 8 and _kernel.matmul_type == "GEMV":
                continue  # skip gemvs for larger batch-sizes
            t.append(eval_time(_kernel.forward, args))
        indx = np.argmin(t)
        GEMLITE_TRITON_CACHE[signature] = {
            "forward": self.kernels[indx].forward,
            "time": t[indx],
        }

    # Main forward pass
    def forward_auto(self, x):
        global GEMLITE_TRITON_CACHE
        out_shape = x.shape[:-1] + (self.out_features,)
        args = [
            x.view(-1, x.shape[-1]),
            self.W_q,
            self.scales,
            self.zeros,
            self.W_nbits,
            self.group_size,
            self.unpack_mask,
            self.elements_per_sample,
            self.acc_dtype,
        ]

        _signature = (x.shape[0],) + self.signature
        if _signature not in GEMLITE_TRITON_CACHE:
            self.warmup(_signature, args)

        out = GEMLITE_TRITON_CACHE[_signature]["forward"](*args).view(out_shape)

        if self.bias is not None:
            out += self.bias
        return out

    def forward_manual(self, x, matmul_type="GEMM"):
        out_shape = x.shape[:-1] + (self.out_features,)

        out = (
            GEMLITE_TRITON_MAPPING[(self.input_dtype.value, matmul_type)]
            .forward(
                x.view(-1, x.shape[-1]),
                self.W_q,
                self.scales,
                self.zeros,
                self.W_nbits,
                self.group_size,
                self.unpack_mask,
                self.elements_per_sample,
                self.acc_dtype,
            )
            .view(out_shape)
        )

        if self.bias is not None:
            out += self.bias
        return out


###################################################################################################################################
###################################################################################################################################
GemLiteLinear = GemLiteLinearTriton  # Triton by default
