"""Helper functions shared by modules in aqt/jax."""


def normalize_axes(axes, ndim):
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def broadcast_rank(source, target):
  """Broadcasts source to match target's rank following Numpy semantics."""
  return source.reshape((1,) * (target.ndim - source.ndim) + source.shape)
