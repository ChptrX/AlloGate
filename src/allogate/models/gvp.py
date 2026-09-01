"""A paper-specified geometric vector perceptron for AlloGate.

This module is newly organized around the equations in Jing et al., ICLR 2021.
Known MIT-licensed implementations in the project's provenance chain are
credited in THIRD_PARTY_NOTICES.md. This file does not copy their source text.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ChannelDimensions:
    """Numbers of invariant scalar and equivariant vector channels."""

    scalars: int
    vectors: int

    def __post_init__(self) -> None:
        if self.scalars < 0 or self.vectors < 0:
            raise ValueError("channel dimensions cannot be negative")
        if self.scalars + self.vectors == 0:
            raise ValueError("at least one scalar or vector channel is required")


ScalarVector = tuple[Tensor, Tensor]


def _check_pair(value: ScalarVector, expected: ChannelDimensions) -> None:
    scalars, vectors = value
    if scalars.ndim < 1 or vectors.ndim < 2:
        raise ValueError("scalar/vector tensors must have shapes (..., S) and (..., V, 3)")
    if scalars.shape[:-1] != vectors.shape[:-2]:
        raise ValueError("scalar and vector tensors must share their batch dimensions")
    if vectors.shape[-1] != 3:
        raise ValueError("the vector coordinate axis must have length three")
    if scalars.shape[-1] != expected.scalars or vectors.shape[-2] != expected.vectors:
        raise ValueError(
            f"expected ({expected.scalars}, {expected.vectors}) channels, "
            f"received ({scalars.shape[-1]}, {vectors.shape[-2]})"
        )
    if scalars.device != vectors.device:
        raise ValueError("scalar and vector tensors must be on the same device")


def vector_lengths(vectors: Tensor, *, epsilon: float = 1.0e-8) -> Tensor:
    """Return stable Euclidean lengths for the final coordinate axis."""

    if vectors.shape[-1] != 3:
        raise ValueError("the vector coordinate axis must have length three")
    compute_dtype = torch.float64 if vectors.dtype == torch.float64 else torch.float32
    squared = torch.sum(vectors.to(compute_dtype) ** 2, dim=-1)
    return torch.sqrt(torch.clamp(squared, min=epsilon * epsilon)).to(vectors.dtype)


class GeometricVectorPerceptron(nn.Module):
    """Map scalar/vector features while preserving three-dimensional equivariance.

    Vector channels are mixed only by coefficients that are independent of the
    xyz coordinate axis. Vector lengths are the sole vector-to-scalar signal,
    and scalar-derived gates rescale whole vectors. These constraints make the
    scalar output rotation invariant and the vector output rotation equivariant.
    """

    def __init__(
        self,
        input_dimensions: ChannelDimensions,
        output_dimensions: ChannelDimensions,
        *,
        scalar_activation: Callable[[Tensor], Tensor] | None = F.silu,
        gate_vectors: bool = True,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.scalar_activation = scalar_activation
        self.gate_vectors = gate_vectors
        self.epsilon = epsilon

        hidden_vectors = max(input_dimensions.vectors, output_dimensions.vectors)
        self.hidden_vector_channels = hidden_vectors

        if input_dimensions.vectors and hidden_vectors:
            self.vector_projection = nn.Parameter(torch.empty(hidden_vectors, input_dimensions.vectors))
        else:
            self.register_parameter("vector_projection", None)

        scalar_inputs = input_dimensions.scalars + hidden_vectors
        self.scalar_weight = nn.Parameter(torch.empty(output_dimensions.scalars, scalar_inputs))
        self.scalar_bias = nn.Parameter(torch.empty(output_dimensions.scalars))

        if output_dimensions.vectors and hidden_vectors:
            self.vector_output_weight = nn.Parameter(torch.empty(output_dimensions.vectors, hidden_vectors))
        else:
            self.register_parameter("vector_output_weight", None)

        if output_dimensions.vectors and gate_vectors:
            self.gate_weight = nn.Parameter(torch.empty(output_dimensions.vectors, output_dimensions.scalars))
            self.gate_bias = nn.Parameter(torch.empty(output_dimensions.vectors))
        else:
            self.register_parameter("gate_weight", None)
            self.register_parameter("gate_bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in (self.vector_projection, self.scalar_weight, self.vector_output_weight, self.gate_weight):
            if parameter is not None:
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        fan_in = self.scalar_weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in) if fan_in else 0.0
        nn.init.uniform_(self.scalar_bias, -bound, bound)
        if self.gate_bias is not None:
            gate_fan_in = self.gate_weight.shape[1]
            gate_bound = 1.0 / math.sqrt(gate_fan_in) if gate_fan_in else 0.0
            nn.init.uniform_(self.gate_bias, -gate_bound, gate_bound)

    def _hidden_vectors(self, vectors: Tensor, scalar_batch_shape: torch.Size) -> Tensor:
        if self.vector_projection is None:
            return vectors.new_zeros((*scalar_batch_shape, self.hidden_vector_channels, 3))
        return torch.einsum("...ic,hi->...hc", vectors, self.vector_projection)

    def forward(self, value: ScalarVector) -> ScalarVector:
        _check_pair(value, self.input_dimensions)
        scalars, vectors = value
        hidden_vectors = self._hidden_vectors(vectors, scalars.shape[:-1])
        invariant_lengths = vector_lengths(hidden_vectors, epsilon=self.epsilon).to(scalars.dtype)
        scalar_features = torch.cat((scalars, invariant_lengths), dim=-1)
        output_scalars = F.linear(scalar_features, self.scalar_weight, self.scalar_bias)
        if self.scalar_activation is not None:
            output_scalars = self.scalar_activation(output_scalars)

        if self.vector_output_weight is None:
            output_vectors = vectors.new_zeros((*scalars.shape[:-1], self.output_dimensions.vectors, 3))
        else:
            output_vectors = torch.einsum("...hc,oh->...oc", hidden_vectors, self.vector_output_weight)
            if self.gate_weight is not None:
                gates = torch.sigmoid(F.linear(output_scalars, self.gate_weight, self.gate_bias))
                output_vectors = output_vectors * gates.unsqueeze(-1)
        return output_scalars, output_vectors
