"""Shape utility functions used by the models."""

from typing import Sequence


def assert_shapes_equal(shape: Sequence[int],
                        expected_shape: Sequence[int]) -> None:
  """Check if the shape of a tensor exactly matches an expected shape."""

  assert shape == expected_shape, (f"Expected tensor to have shape "
                                   f"{expected_shape} but got {shape}.")


def assert_shapes_compatible(lhs: Sequence[int], rhs: Sequence[int]):
  """Check if the shape of two tensors are broadcast-compatible."""

  # This is a stricter-than-necessary check for broadcast-comptability,
  # but it's error-prone to allow broadcasting to insert new dimensions so
  # we don't allow that.
  if len(lhs) != len(rhs):
    return False

  for lhs_dim, rhs_dim in zip(lhs, rhs):
    # A dimension of 1 will be broadcasted to match the other tensor's dimension
    if lhs_dim == 1 or rhs_dim == 1:
      continue
    assert lhs_dim == rhs_dim, (f"Tensors with shape {lhs} and {rhs} cannot "
                                f"be broadcasted together")
