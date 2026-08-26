// f64 table-driven exp/log normal-lane cores.
//
// The range-reduction algorithms and constants are derived from Arm Optimized Routines
//
// Arm's AdvSIMD reference uses N=128 segments: exp uses a scale table and degree-4
// exp(r)-1 polynomial; log uses {1/c, log(c)} table entries and degree-5 log1p(r)-r.
// This HYLE form preserves that arithmetic across AVX2, AVX-512 and NEON. Exceptional
// lanes are deliberately excluded by the callers and handled by scalar libm there.
#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/core/compiler.h"
#include "hyle/simd/vec.h"

namespace hyle::simd::detail::table_f64 {
using namespace hyle::simd;

inline constexpr std::size_t kTableBits = 7;
inline constexpr std::size_t kTableSize = std::size_t{1} << kTableBits;
inline constexpr std::uint64_t kTableMask = kTableSize - 1;

alignas(64) inline constexpr std::uint64_t kExpScaleBits[kTableSize] = {
  0x3ff0000000000000ULL, 0x3feff63da9fb3335ULL, 0x3fefec9a3e778061ULL, 0x3fefe315e86e7f85ULL,
  0x3fefd9b0d3158574ULL, 0x3fefd06b29ddf6deULL, 0x3fefc74518759bc8ULL, 0x3fefbe3ecac6f383ULL,
  0x3fefb5586cf9890fULL, 0x3fefac922b7247f7ULL, 0x3fefa3ec32d3d1a2ULL, 0x3fef9b66affed31bULL,
  0x3fef9301d0125b51ULL, 0x3fef8abdc06c31ccULL, 0x3fef829aaea92de0ULL, 0x3fef7a98c8a58e51ULL,
  0x3fef72b83c7d517bULL, 0x3fef6af9388c8deaULL, 0x3fef635beb6fcb75ULL, 0x3fef5be084045cd4ULL,
  0x3fef54873168b9aaULL, 0x3fef4d5022fcd91dULL, 0x3fef463b88628cd6ULL, 0x3fef3f49917ddc96ULL,
  0x3fef387a6e756238ULL, 0x3fef31ce4fb2a63fULL, 0x3fef2b4565e27cddULL, 0x3fef24dfe1f56381ULL,
  0x3fef1e9df51fdee1ULL, 0x3fef187fd0dad990ULL, 0x3fef1285a6e4030bULL, 0x3fef0cafa93e2f56ULL,
  0x3fef06fe0a31b715ULL, 0x3fef0170fc4cd831ULL, 0x3feefc08b26416ffULL, 0x3feef6c55f929ff1ULL,
  0x3feef1a7373aa9cbULL, 0x3feeecae6d05d866ULL, 0x3feee7db34e59ff7ULL, 0x3feee32dc313a8e5ULL,
  0x3feedea64c123422ULL, 0x3feeda4504ac801cULL, 0x3feed60a21f72e2aULL, 0x3feed1f5d950a897ULL,
  0x3feece086061892dULL, 0x3feeca41ed1d0057ULL, 0x3feec6a2b5c13cd0ULL, 0x3feec32af0d7d3deULL,
  0x3feebfdad5362a27ULL, 0x3feebcb299fddd0dULL, 0x3feeb9b2769d2ca7ULL, 0x3feeb6daa2cf6642ULL,
  0x3feeb42b569d4f82ULL, 0x3feeb1a4ca5d920fULL, 0x3feeaf4736b527daULL, 0x3feead12d497c7fdULL,
  0x3feeab07dd485429ULL, 0x3feea9268a5946b7ULL, 0x3feea76f15ad2148ULL, 0x3feea5e1b976dc09ULL,
  0x3feea47eb03a5585ULL, 0x3feea34634ccc320ULL, 0x3feea23882552225ULL, 0x3feea155d44ca973ULL,
  0x3feea09e667f3bcdULL, 0x3feea012750bdabfULL, 0x3fee9fb23c651a2fULL, 0x3fee9f7df9519484ULL,
  0x3fee9f75e8ec5f74ULL, 0x3fee9f9a48a58174ULL, 0x3fee9feb564267c9ULL, 0x3feea0694fde5d3fULL,
  0x3feea11473eb0187ULL, 0x3feea1ed0130c132ULL, 0x3feea2f336cf4e62ULL, 0x3feea427543e1a12ULL,
  0x3feea589994cce13ULL, 0x3feea71a4623c7adULL, 0x3feea8d99b4492edULL, 0x3feeaac7d98a6699ULL,
  0x3feeace5422aa0dbULL, 0x3feeaf3216b5448cULL, 0x3feeb1ae99157736ULL, 0x3feeb45b0b91ffc6ULL,
  0x3feeb737b0cdc5e5ULL, 0x3feeba44cbc8520fULL, 0x3feebd829fde4e50ULL, 0x3feec0f170ca07baULL,
  0x3feec49182a3f090ULL, 0x3feec86319e32323ULL, 0x3feecc667b5de565ULL, 0x3feed09bec4a2d33ULL,
  0x3feed503b23e255dULL, 0x3feed99e1330b358ULL, 0x3feede6b5579fdbfULL, 0x3feee36bbfd3f37aULL,
  0x3feee89f995ad3adULL, 0x3feeee07298db666ULL, 0x3feef3a2b84f15fbULL, 0x3feef9728de5593aULL,
  0x3feeff76f2fb5e47ULL, 0x3fef05b030a1064aULL, 0x3fef0c1e904bc1d2ULL, 0x3fef12c25bd71e09ULL,
  0x3fef199bdd85529cULL, 0x3fef20ab5fffd07aULL, 0x3fef27f12e57d14bULL, 0x3fef2f6d9406e7b5ULL,
  0x3fef3720dcef9069ULL, 0x3fef3f0b555dc3faULL, 0x3fef472d4a07897cULL, 0x3fef4f87080d89f2ULL,
  0x3fef5818dcfba487ULL, 0x3fef60e316c98398ULL, 0x3fef69e603db3285ULL, 0x3fef7321f301b460ULL,
  0x3fef7c97337b9b5fULL, 0x3fef864614f5a129ULL, 0x3fef902ee78b3ff6ULL, 0x3fef9a51fbc74c83ULL,
  0x3fefa4afa2a490daULL, 0x3fefaf482d8e67f1ULL, 0x3fefba1bee615a27ULL, 0x3fefc52b376bba97ULL,
  0x3fefd0765b6e4540ULL, 0x3fefdbfdad9cbe14ULL, 0x3fefe7c1819e90d8ULL, 0x3feff3c22b8f71f1ULL,
};

alignas(64) inline constexpr double kLogInvC[kTableSize] = {
  0x1.6a133d0dec120p+0, 0x1.6815f2f3e42edp+0, 0x1.661e39be1ac9ep+0, 0x1.642bfa30ac371p+0,
  0x1.623f1d916f323p+0, 0x1.60578da220f65p+0, 0x1.5e75349dea571p+0, 0x1.5c97fd387a75ap+0,
  0x1.5abfd2981f200p+0, 0x1.58eca051dc99cp+0, 0x1.571e526d9df12p+0, 0x1.5554d555b3fcbp+0,
  0x1.539015e2a20cdp+0, 0x1.51d0014ee0164p+0, 0x1.50148538cd9eep+0, 0x1.4e5d8f9f698a1p+0,
  0x1.4cab0edca66bep+0, 0x1.4afcf1a9db874p+0, 0x1.495327136e16fp+0, 0x1.47ad9e84af28fp+0,
  0x1.460c47b39ae15p+0, 0x1.446f12b278001p+0, 0x1.42d5efdd720ecp+0, 0x1.4140cfe001a0fp+0,
  0x1.3fafa3b421f69p+0, 0x1.3e225c9c8ece5p+0, 0x1.3c98ec29a211ap+0, 0x1.3b13442a413fep+0,
  0x1.399156baa3c54p+0, 0x1.38131639b4cdbp+0, 0x1.36987540fbf53p+0, 0x1.352166b648f61p+0,
  0x1.33adddb3eb575p+0, 0x1.323dcd99fc1d3p+0, 0x1.30d129fefc7d2p+0, 0x1.2f67e6b72fe7dp+0,
  0x1.2e01f7cf8b187p+0, 0x1.2c9f518ddc86ep+0, 0x1.2b3fe86e5f413p+0, 0x1.29e3b1211b25cp+0,
  0x1.288aa08b373cfp+0, 0x1.2734abcaa8467p+0, 0x1.25e1c82459b81p+0, 0x1.2491eb1ad59c5p+0,
  0x1.23450a54048b5p+0, 0x1.21fb1bb09e578p+0, 0x1.20b415346d8f7p+0, 0x1.1f6fed179a1acp+0,
  0x1.1e2e99b93c7b3p+0, 0x1.1cf011a7a882ap+0, 0x1.1bb44b97dba5ap+0, 0x1.1a7b3e66cdd4fp+0,
  0x1.1944e11dc56cdp+0, 0x1.18112aebb1a6ep+0, 0x1.16e013231b7e9p+0, 0x1.15b1913f156cfp+0,
  0x1.14859cdedde13p+0, 0x1.135c2dc68cfa4p+0, 0x1.12353bdb01684p+0, 0x1.1110bf25b85b4p+0,
  0x1.0feeafd2f8577p+0, 0x1.0ecf062c51c3bp+0, 0x1.0db1baa076c8bp+0, 0x1.0c96c5bb3048ep+0,
  0x1.0b7e20263e070p+0, 0x1.0a67c2acd0ce3p+0, 0x1.0953a6391e982p+0, 0x1.0841c3caea380p+0,
  0x1.07321489b13eap+0, 0x1.062491aee9904p+0, 0x1.05193497a7cc5p+0, 0x1.040ff6b5f5e9fp+0,
  0x1.0308d19aa6127p+0, 0x1.0203beedb0c67p+0, 0x1.010037d38bcc2p+0, 1.0,
  0x1.fc06d493cca10p-1, 0x1.f81e6ac3b918fp-1, 0x1.f44546ef18996p-1, 0x1.f07b10382c84bp-1,
  0x1.ecbf7070e59d4p-1, 0x1.e91213f715939p-1, 0x1.e572a9a75f7b7p-1, 0x1.e1e0e2c530207p-1,
  0x1.de5c72d8a8be3p-1, 0x1.dae50fa5658ccp-1, 0x1.d77a71145a2dap-1, 0x1.d41c51166623ep-1,
  0x1.d0ca6ba0bb29fp-1, 0x1.cd847e8e59681p-1, 0x1.ca4a499693e00p-1, 0x1.c71b8e399e821p-1,
  0x1.c3f80faf19077p-1, 0x1.c0df92dc2b0ecp-1, 0x1.bdd1de3cbb542p-1, 0x1.baceb9e1007a3p-1,
  0x1.b7d5ef543e55ep-1, 0x1.b4e749977d953p-1, 0x1.b20295155478ep-1, 0x1.af279f8e82be2p-1,
  0x1.ac5638197fdf3p-1, 0x1.a98e2f102e087p-1, 0x1.a6cf5606d05c1p-1, 0x1.a4197fc04d746p-1,
  0x1.a16c80293dc01p-1, 0x1.9ec82c4dc5bc9p-1, 0x1.9c2c5a491f534p-1, 0x1.9998e1480b618p-1,
  0x1.970d9977c6c2dp-1, 0x1.948a5c023d212p-1, 0x1.920f0303d6809p-1, 0x1.8f9b698a98b45p-1,
  0x1.8d2f6b81726f6p-1, 0x1.8acae5bb55badp-1, 0x1.886db5d9275b8p-1, 0x1.8617ba567c13cp-1,
  0x1.83c8d27487800p-1, 0x1.8180de3c5dbe7p-1, 0x1.7f3fbe71cdb71p-1, 0x1.7d055498071c1p-1,
  0x1.7ad182e54f65ap-1, 0x1.78a42c3c90125p-1, 0x1.767d342f76944p-1, 0x1.745c7ef26b00ap-1,
  0x1.7241f15769d0fp-1, 0x1.702d70d396e41p-1, 0x1.6e1ee3700cd11p-1, 0x1.6c162fc9cbe02p-1,
};

alignas(64) inline constexpr double kLogC[kTableSize] = {
  -0x1.62fe995eb963ap-2, -0x1.5d5a48dad6b67p-2, -0x1.57bde257d2769p-2, -0x1.52294fbf2af55p-2,
  -0x1.4c9c7b598aa38p-2, -0x1.47174fc5ff560p-2, -0x1.4199b7fa7b5cap-2, -0x1.3c239f48cfb99p-2,
  -0x1.36b4f154d2aebp-2, -0x1.314d9a0ff32fbp-2, -0x1.2bed85cca3cffp-2, -0x1.2694a11421af9p-2,
  -0x1.2142d8d014fb2p-2, -0x1.1bf81a2c77776p-2, -0x1.16b452a39c6a4p-2, -0x1.11776ffa6c67ep-2,
  -0x1.0c416035020e0p-2, -0x1.071211aa10fdap-2, -0x1.01e972e293b1bp-2, -0x1.f98ee587fd434p-3,
  -0x1.ef5800ad716fbp-3, -0x1.e52e160484698p-3, -0x1.db1104b19352ep-3, -0x1.d100ac59e0bd6p-3,
  -0x1.c6fced287c3bdp-3, -0x1.bd05a7b317c29p-3, -0x1.b31abd229164fp-3, -0x1.a93c0edadb0a3p-3,
  -0x1.9f697ee30d7ddp-3, -0x1.95a2efa9aa40ap-3, -0x1.8be843d796044p-3, -0x1.82395ecc477edp-3,
  -0x1.7896240966422p-3, -0x1.6efe77aca8c55p-3, -0x1.65723e117ec5cp-3, -0x1.5bf15c0955706p-3,
  -0x1.527bb6c111da1p-3, -0x1.491133c939f8fp-3, -0x1.3fb1b90c7fc58p-3, -0x1.365d2cc485f8p-3,
  -0x1.2d13758970de7p-3, -0x1.23d47a721fd47p-3, -0x1.1aa0229f25ec2p-3, -0x1.117655ddebc3bp-3,
  -0x1.0856fbf83ab6bp-3, -0x1.fe83fabbaa106p-4, -0x1.ec6e8507a56cdp-4, -0x1.da6d68c7cc2eap-4,
  -0x1.c88078462be0cp-4, -0x1.b6a786a423565p-4, -0x1.a4e2676ac7f85p-4, -0x1.9330eea777e76p-4,
  -0x1.8192f134d5ad9p-4, -0x1.70084464f0538p-4, -0x1.5e90bdec5cb1fp-4, -0x1.4d2c3433c5536p-4,
  -0x1.3bda7e219879ap-4, -0x1.2a9b732d27194p-4, -0x1.196eeb2b10807p-4, -0x1.0854be8ef8a7ep-4,
  -0x1.ee998cb277432p-5, -0x1.ccadb79919fb9p-5, -0x1.aae5b1d8618b0p-5, -0x1.89413015d7442p-5,
  -0x1.67bfe7bf158dep-5, -0x1.46618f83941bep-5, -0x1.2525df1b0618ap-5, -0x1.040c8e2f77c6ap-5,
  -0x1.c62aad39f738ap-6, -0x1.847fe3bdead9cp-6, -0x1.43183683400acp-6, -0x1.01f31c4e1d544p-6,
  -0x1.82201d1e6b69ap-7, -0x1.00dd0f3e1bfd6p-7, -0x1.ff6fe1feb4e53p-9, 0.0,
  0x1.fe91885ec8e20p-8, 0x1.fc516f716296dp-7, 0x1.7bb4dd70a015bp-6, 0x1.f84c99b34b674p-6,
  0x1.39f9ce4fb2d71p-5, 0x1.7756c0fd22e78p-5, 0x1.b43ee82db8f3ap-5, 0x1.f0b3fced60034p-5,
  0x1.165bd78d4878ep-4, 0x1.3425d2715ebe6p-4, 0x1.51b8bd91b7915p-4, 0x1.6f15632c76a47p-4,
  0x1.8c3c88ecbe503p-4, 0x1.a92ef077625dap-4, 0x1.c5ed5745fa006p-4, 0x1.e27876de1c993p-4,
  0x1.fed104fce4cdcp-4, 0x1.0d7bd9c17d78bp-3, 0x1.1b76986cef97bp-3, 0x1.295913d24f750p-3,
  0x1.37239fa295d17p-3, 0x1.44d68dd78714bp-3, 0x1.52722ebe5d780p-3, 0x1.5ff6d12671f98p-3,
  0x1.6d64c2389484bp-3, 0x1.7abc4da40fddap-3, 0x1.87fdbda1e8452p-3, 0x1.95295b06a5f37p-3,
  0x1.a23f6d34abbc5p-3, 0x1.af403a28e04f2p-3, 0x1.bc2c06a85721ap-3, 0x1.c903161240163p-3,
  0x1.d5c5aa93287ebp-3, 0x1.e274051823fa9p-3, 0x1.ef0e656300c16p-3, 0x1.fb9509f05aa2ap-3,
  0x1.04041821f37afp-2, 0x1.0a340a49b3029p-2, 0x1.105a7918a126dp-2, 0x1.1677819812b84p-2,
  0x1.1c8b405b40c0ep-2, 0x1.2295d16cfa6b1p-2, 0x1.28975066318a2p-2, 0x1.2e8fd855d86fcp-2,
  0x1.347f83d605e59p-2, 0x1.3a666d1244588p-2, 0x1.4044adb6f8ec4p-2, 0x1.461a5f077558cp-2,
  0x1.4be799e20b9c8p-2, 0x1.51ac76a6b79dfp-2, 0x1.57690d5744a45p-2, 0x1.5d1d758e45217p-2,
};

template <class D, class V>
HYLE_FORCE_INLINE V ExpNormal(D d, V x) noexcept {
  const auto du64 = Rebind<std::uint64_t, D>();
  const auto di64 = Rebind<std::int64_t, D>();
  const auto shift = Set(d, 0x1.8p52);

  // n = round(x * N/ln2); its low kTableBits select 2^(j/N).
  const V z = MulAdd(x, Set(d, 0x1.71547652b82fep7), shift);
  const V n = Sub(z, shift);
  const auto bits = BitCast(du64, z);
  const auto index = BitCast(
      di64, And(bits, Set(du64, static_cast<std::uint64_t>(kTableMask))));

  V r = MulAdd(n, Set(d, -0x1.62e42fefa39efp-8), x);
  r = MulAdd(n, Set(d, -0x1.abc9e3b39803f3p-63), r);

  // exp(r)-1 = r + r^2 * (c0 + c1*r + c2*r^2).
  const V r2 = Mul(r, r);
  V poly = MulAdd(Set(d, 0x1.55555c75adbb2p-3), r,
                       Set(d, 0x1.ffffffffffd43p-2));
  poly = MulAdd(Set(d, 0x1.55555da646206p-5), r2, poly);
  poly = MulAdd(poly, r2, r);

  const auto scale_bits = Add(GatherIndex(du64, kExpScaleBits, index),
                                  ShiftLeft<52 - kTableBits>(bits));
  const V scale = BitCast(d, scale_bits);
  return MulAdd(scale, poly, scale);
}

template <class D, class V>
HYLE_FORCE_INLINE V LogNormal(D d, V x) noexcept {
  const auto du64 = Rebind<std::uint64_t, D>();
  const auto di64 = Rebind<std::int64_t, D>();
  const auto off = Set(du64, 0x3fe6900900000000ULL);
  const auto sign_exp = Set(du64, 0xfff0000000000000ULL);
  const auto bits = BitCast(du64, x);

  // x = 2^k z, z is exact in [off, 2*off); table index chooses c near z.
  const auto tmp = Sub(bits, off);
  const auto index = BitCast(
      di64, And(ShiftRight<52 - kTableBits>(tmp),
                    Set(du64, static_cast<std::uint64_t>(kTableMask))));
  const auto zbits = Sub(bits, And(tmp, sign_exp));
  const V z = BitCast(d, zbits);
  const V invc = GatherIndex(d, kLogInvC, index);
  const V logc = GatherIndex(d, kLogC, index);
  const V kd = ConvertTo(d, ShiftRight<52>(BitCast(di64, tmp)));
  const V ln2 = Set(d, 0x1.62e42fefa39efp-1);

  const V r = MulAdd(z, invc, Set(d, -1.0));
  const V hi = MulAdd(kd, ln2, Add(logc, r));
  const V r2 = Mul(r, r);

  // log1p(r)-r = r^2 * (c0 + c1*r + r^2*(c2 + c3*r + c4*r^2)).
  const V p = MulAdd(r, Set(d, 0x1.55555555170d4p-2),
                          Set(d, -0x1.ffffffffffff7p-2));
  V q = MulAdd(r, Set(d, 0x1.999b2e90e94cap-3),
                   Set(d, -0x1.0000000399c27p-2));
  q = MulAdd(r2, Set(d, -0x1.554e550bd501ep-3), q);
  q = MulAdd(q, r2, p);
  return MulAdd(q, r2, hi);
}

}  // namespace hyle::simd::detail::table_f64
