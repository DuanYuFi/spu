"""Plaintext specification for bounded F61 PPML, never a protocol backend.

No function here provides privacy, authentication or malicious security. Python
integers make this an independent oracle for C++ overflow and representation
tests. Secure implementations must replace each secret-dependent branch/open.
"""

P = (1 << 61) - 1
HALF = P // 2
WORD_BITS = 64
WORD_MASK = (1 << WORD_BITS) - 1


def centered(residue, prime=P):
    if not 0 <= residue < prime:
        raise ValueError("noncanonical residue")
    return residue if residue <= prime // 2 else residue - prime


def encode_integer(value, prime=P):
    if not -(prime // 2) <= value <= prime // 2:
        raise ValueError("value outside the centered range")
    return value % prime


def residue_bits_from_mask(value, mask, prime=P):
    """KnownPrime full-decomposition contract, including the wrap correction.

    c=(value-mask) mod p is the only input-dependent opened value in this
    arithmetic specification. In MPC, mask is uniform, secret and correlated
    with validated binary bits. The additions and comparison below are secret
    binary circuits. This function is a plaintext oracle, not those circuits.
    """
    if not (0 <= value < prime and 0 <= mask < prime):
        raise ValueError("noncanonical operand")
    opened = (value - mask) % prime
    total = opened + mask
    result = total - (prime if total >= prime else 0)
    return [(result >> bit) & 1 for bit in range(prime.bit_length())]


def a2b(value, prime=P):
    word = centered(value, prime) & WORD_MASK
    return [(word >> bit) & 1 for bit in range(WORD_BITS)]


def b2a(bits, prime=P):
    if not 1 <= len(bits) <= WORD_BITS or any(b not in (0, 1) for b in bits):
        raise ValueError("expected 1..64 bits")
    value = sum(bit << i for i, bit in enumerate(bits))
    if len(bits) == WORD_BITS:
        value -= bits[-1] << WORD_BITS
    return value % prime


def trunc(value, shift, prime=P):
    if not 0 <= shift < WORD_BITS:
        raise ValueError("invalid shift")
    return (centered(value, prime) // (1 << shift)) % prime


class BoundedFixedPoint:
    """Reference arithmetic with explicit range checks and recorded maxima."""

    def __init__(self, fraction_bits=16):
        if not 1 <= fraction_bits < 30:
            raise ValueError("unsupported fraction width")
        self.fraction_bits = fraction_bits
        self.max_abs = 0
        self.max_product = 0
        self.max_accumulator = 0

    def checked(self, value):
        encode_integer(value)
        self.max_abs = max(self.max_abs, abs(value))
        return value

    def multiply(self, x, y):
        product = self.checked(x * y)
        self.max_product = max(self.max_product, abs(product))
        return self.checked(product // (1 << self.fraction_bits))

    def dot(self, x, y):
        if len(x) != len(y):
            raise ValueError("shape mismatch")
        accumulator = 0
        for a, b in zip(x, y):
            product = self.checked(a * b)
            self.max_product = max(self.max_product, abs(product))
            accumulator = self.checked(accumulator + product)
            self.max_accumulator = max(self.max_accumulator, abs(accumulator))
        return self.checked(accumulator // (1 << self.fraction_bits))
