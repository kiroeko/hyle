// Plain storage conversion matches the public SIMD E4M3FN contract. Integer
// fields retain NaN/zero signs and avoid FTZ or arithmetic at rounding ties.
uint fp8_e4m3_to_f32_bits(uint code) {
    uint sign_bit = (code & 128u) << 24u;
    uint magnitude = code & 127u;
    uint exponent = magnitude >> 3u, mantissa = magnitude & 7u;
    if (magnitude == 127u) return sign_bit | 0x7fc00000u;
    if (exponent != 0u)
        return sign_bit | ((exponent + 120u) << 23u) | (mantissa << 20u);
    if (mantissa == 0u) return sign_bit;
    uint leading = uint(findMSB(mantissa));
    return sign_bit | ((leading + 118u) << 23u) |
           ((mantissa << (23u - leading)) & 0x7fffffu);
}
uint f32_to_fp8_e4m3_bits(uint bits) {
    uint sign_bit = (bits >> 24u) & 128u;
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return sign_bit | 127u;
    if (magnitude >= 0x43e00000u) return sign_bit | 126u;
    int exponent = int(magnitude >> 23u) - 120;
    uint high, remainder, midpoint;
    if (exponent >= 1) {
        high = (uint(exponent) << 3u) | ((magnitude >> 20u) & 7u);
        remainder = magnitude & 0xfffffu;
        midpoint = 0x80000u;
    } else {
        uint shift = uint(1 - exponent);
        if (shift > 4u) return sign_bit;
        uint significand = (magnitude & 0x7fffffu) | 0x800000u;
        uint drop = 20u + shift;
        high = significand >> drop;
        remainder = significand & ((1u << drop) - 1u);
        midpoint = 1u << (drop - 1u);
    }
    high += uint(remainder > midpoint || (remainder == midpoint && (high & 1u) != 0u));
    return sign_bit | min(high, 126u);
}
