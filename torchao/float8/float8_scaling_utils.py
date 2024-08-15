# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for scaling high precision tensors to float8.
"""

from typing import Optional, Tuple, Literal

import torch

from torchao.float8.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    hp_tensor_and_scale_to_float8,
    LinearMMConfig,
    ScaledMMConfig,
    tensor_already_casted_to_fp8,
)

from torchao.float8.float8_utils import (
    amax_history_to_scale,
    e4m3_dtype,
    e5m2_dtype,
    tensor_to_amax,
    tensor_to_scale,
)


def hp_tensor_to_float8_dynamic(
    hp_tensor: torch.Tensor,
    float8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    reduce_amax: bool = False,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
    tile_size: Optional[Tuple[int, int]] = None,
) -> Float8Tensor:
    """
    Given a high precision tensor `hp_tensor`,
    scales `hp_tensor` dynamically and returns a `Float8Tensor` of the result.

    Args:
        hp_tensor: the tensor to convert
        float8_dtype: the float8 dtype to use
        linear_mm_config: Defines the configuration for the scaled_mm for
          the 3 fwd/bwd gemms of linear
        reduce_amax: whether to reduce the max(abs(hp_tensor)) value across distributed ranks
        gemm_input_role: Defines the role of this tensor (input, weight or grad_output) in
          the 3 fwd/bwd gemms of linear
        tile_size | None: This shape of the tile used to calculate the abs max of the input tensor.
            Example: hp_tensor.shape = (M, K)
            - (M, N): The tile is the size of the tensor, i.e. the max is calculated for the entire tensor | TensorWise scaling
            - (1, M); 1 scale per row | RowWise scaling
            - (A, B); where A < M and B < K, 1 scale per (A, B) block | BlockWise scaling
    """
    if tensor_already_casted_to_fp8(hp_tensor):
        return hp_tensor
    scale = tensor_to_scale(hp_tensor, float8_dtype, tile_size, reduce_amax)
    return hp_tensor_and_scale_to_float8(
        hp_tensor,
        scale,
        float8_dtype,
        linear_mm_config,
        gemm_input_role,
    )


def hp_tensor_to_float8_delayed(
    hp_tensor: torch.Tensor,
    s: torch.Tensor,
    float8_dtype: torch.dtype,
    amax_buffer: torch.Tensor,
    linear_mm_config: Optional[LinearMMConfig] = None,
    gemm_input_role: Optional[GemmInputRole] = GemmInputRole.INPUT,
    tile_size: Optional[Tuple[int, int]] = None,
) -> Float8Tensor:
    """
    Given a high precision tensor `hp_tensor` and relevant metadata, scales it using
    delayed scaling and returns a `Float8Tensor` of the result. Specifically:
    1. calculates max(abs(hp_tensor)) and stores the result in `amax_buffer`, inplace
    2. scales `hp_tensor` by `s` and returns the result wrapped in Float8Tensor

    Args:
        hp_tensor: the tensor to convert
        s: the scale to use to convert the tensor
        float8_dtype: the float8 dtype to use
        amax_buffer: the buffer to modify inplace with max(abs(hp_tensor))
        linear_mm_config: Defines the configuration for the scaled_mm for
          the 3 fwd/bwd gemms of linear
        gemm_input_role: Defines the role of this tensor (input, weight or grad_output) in
          the 3 fwd/bwd gemms of linear
        tile_size: This shape of the tile used to calculate the abs max of the input tensor.
            Example: hp_tensor.shape = (M, K)
            - (M, N) | None: The tile is the size of the tensor, i.e. the max is calculated for the entire tensor | TensorWise scaling
            - (1, M); 1 scale per row | RowWise scaling
            - (A, B); where A < M and B < K, 1 scale per (A, B) block | BlockWise scaling
    """
    assert tile_size is None, f"Delayed scaling is not currently supported for non TensorWise scaling, got tile_size {tile_size}"
    amax_buffer.fill_(tensor_to_amax(hp_tensor, tile_size))
    return hp_tensor_and_scale_to_float8(
        hp_tensor,
        s,
        float8_dtype,
        linear_mm_config,
        gemm_input_role,
    )


def _maybe_initialize_amaxes_scales_for_float8_cast(
    x: torch.Tensor,
    cur_amax: torch.Tensor,
    amax_history: torch.Tensor,
    scale: torch.Tensor,
    scale_fn_name:  Literal["max"],
    float8_dtype: torch.dtype,
    is_initialized: bool,
    reduce_amax: bool,
    tile_size: Optional[Tuple[int, int]]
):
    """
    If x is about to be cast to `float8` and the amax buffers are not initialized,
    initializes them inplace.
    """
    assert tile_size is None, f"Delayed scaling is not currently supported for non-TensorWise scaling. Got tile_size: {tile_size}"
    if is_initialized:
        return
    with torch.no_grad():
        # Note: we need to enable distributed reduction here in order
        # to match numerics between single GPU and multi GPU code for
        # activations and gradients

        # We hardcode TensorWise Scaling on delayed scaling for now
        new_amax = tensor_to_amax(x, tile_size=tile_size, reduce_amax=reduce_amax)
        cur_amax.fill_(new_amax)
        amax_history[0] = new_amax
        new_scale = amax_history_to_scale(
            amax_history, float8_dtype, x.dtype, scale_fn_name, stack=False
        )
        scale.copy_(new_scale)


@torch._dynamo.allow_in_graph
class NoopFwToFloat8E5M2BwDelayed(torch.autograd.Function):
    """
    Forward: no-op
    Backward: convert to float8_e5m2 with delayed scaling, initialize if needed
    """

    @staticmethod
    def forward(
        ctx,
        tensor,
        fp8_amax_grad_output,
        fp8_amax_history_grad_output,
        fp8_scale_grad_output,
        scale_fn_name,
        is_amax_initialized,
        linear_mm_config: LinearMMConfig,
    ):
        ctx.save_for_backward(
            fp8_amax_grad_output, fp8_amax_history_grad_output, fp8_scale_grad_output
        )
        ctx.scale_fn_name = scale_fn_name
        ctx.is_amax_initialized = is_amax_initialized
        ctx.linear_mm_config = linear_mm_config
        return tensor

    @staticmethod
    def backward(ctx, go):
        (
            fp8_amax_grad_output,
            fp8_amax_history_grad_output,
            fp8_scale_grad_output,
        ) = ctx.saved_tensors
        scale_fn_name = ctx.scale_fn_name
        is_amax_initialized = ctx.is_amax_initialized

        _maybe_initialize_amaxes_scales_for_float8_cast(
            go,
            fp8_amax_grad_output,
            fp8_amax_history_grad_output,
            fp8_scale_grad_output,
            scale_fn_name,
            e5m2_dtype,
            is_amax_initialized,
            reduce_amax=True,
            tile_size=None,
        )

        fp8_amax_grad_output.fill_(tensor_to_amax(go, None))

        res = hp_tensor_and_scale_to_float8(
            go,
            fp8_scale_grad_output,
            e5m2_dtype,
            ctx.linear_mm_config,
            GemmInputRole.GRAD_OUTPUT,
        )
        empty_grads = None, None, None, None, None, None
        return res, *empty_grads


@torch._dynamo.allow_in_graph
class NoopFwToFloat8E5M2BwDynamic(torch.autograd.Function):
    """
    Forward: no-op
    Backward: convert to float8_e5m2 with dynamic scaling
    """

    @staticmethod
    def forward(
        ctx,
        tensor,
        linear_mm_config: LinearMMConfig,
    ):
        ctx.linear_mm_config = linear_mm_config
        return tensor

    @staticmethod
    def backward(ctx, gradY):
        if tensor_already_casted_to_fp8(gradY):
            return gradY, None
        gradY_scale = tensor_to_scale(gradY, e5m2_dtype, None)
        fp8_tensor = hp_tensor_and_scale_to_float8(
            gradY,
            gradY_scale,
            e5m2_dtype,
            ctx.linear_mm_config,
            GemmInputRole.GRAD_OUTPUT,
        )
        return fp8_tensor, None
