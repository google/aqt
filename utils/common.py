"""General util functions commonly used across different models."""


def get_fp_spec(sig_bit: int, exp_bit: int):
  """Create fp spec which defines precision for floating-point quantization.

  Args:
    sig_bit: the number of bits assigned for significand.
    exp_bit: the number of bits assigned for exponent.

  Returns:
    fp spec
  """
  exp_bound = 2**(exp_bit - 1) - 1
  prec = {'exp_min': -exp_bound, 'exp_max': exp_bound, 'sig_bits': sig_bit}
  return prec
