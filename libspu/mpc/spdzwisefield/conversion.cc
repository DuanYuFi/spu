#include "libspu/mpc/spdzwisefield/conversion.h"

#include <functional>

#include "libspu/core/parallel_utils.h"
#include "libspu/core/platform_utils.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/common/ab_api.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pub2k.h"
#include "libspu/mpc/spdzwisefield/state.h"
#include "libspu/mpc/spdzwisefield/type.h"
#include "libspu/mpc/spdzwisefield/value.h"

#define MYLOG(x) \
  if (comm->getRank() == 0) std::cout << x << std::endl

namespace spu::mpc::spdzwisefield {

template <typename T>
static std::vector<bool> bitDecompose(ArrayView<T> in, size_t nbits) {
  // decompose each bit of an array of element.
  std::vector<bool> dep(in.numel() * nbits);
  pforeach(0, in.numel(), [&](int64_t idx) {
    for (size_t bit = 0; bit < nbits; bit++) {
      size_t flat_idx = idx * nbits + bit;
      dep[flat_idx] = static_cast<bool>((in[idx] >> bit) & 0x1);
    }
  });
  return dep;
}

template <typename T>
static std::vector<T> bitCompose(absl::Span<T const> in, size_t nbits) {
  SPU_ENFORCE(in.size() % nbits == 0);
  std::vector<T> out(in.size() / nbits, 0);
  pforeach(0, out.size(), [&](int64_t idx) {
    for (size_t bit = 0; bit < nbits; bit++) {
      size_t flat_idx = idx * nbits + bit;
      out[idx] += in[flat_idx] << bit;
    }
  });
  return out;
}

template <typename T>
static std::vector<T> open_semi_honest(Object* ctx,
                                       std::vector<std::array<T, 2>> in) {
  auto* comm = ctx->getState<Communicator>();
  using Field = SpdzWiseFieldState::Field;

  std::vector<T> buf(in.size());
  pforeach(0, in.size(), [&](int64_t idx) { buf[idx] = in[idx][1]; });

  std::vector<T> result = comm->rotate<T>(buf, "open_semi_honest");

  pforeach(0, in.size(), [&](int64_t idx) {
    result[idx] = Field::add(result[idx], in[idx][0], in[idx][1]);
  });

  return result;
}

ArrayRef A2B::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  auto* edabit_state = ctx->getState<EdabitState>();
  auto* comm = ctx->getState<Communicator>();

  using Field = SpdzWiseFieldState::Field;

  auto _in = ArrayView<std::array<uint64_t, 4>>(in);
  auto edabits = edabit_state->gen_edabits(ctx->caller(), PT_U64, in.numel());

  std::vector<uint64_t> s_plus_r(in.numel());

  pforeach(0, in.numel(), [&](uint64_t idx) {
    s_plus_r[idx] = Field::add(_in[idx][1], edabits[idx].ashare[1]);
  });

  auto receive = comm->rotate<uint64_t>(s_plus_r, "A2B.open");

  pforeach(0, in.numel(), [&](uint64_t idx) {
    s_plus_r[idx] = Field::add(s_plus_r[idx], receive[idx], _in[idx][0],
                               edabits[idx].ashare[0]);
  });

  ArrayRef binary_public(makeType<Pub2kTy>(FM64), in.numel());
  auto _binary_public = ArrayView<uint64_t>(binary_public);

  pforeach(0, in.numel(),
           [&](uint64_t idx) { _binary_public[idx] = s_plus_r[idx]; });

  ArrayRef binary_share = ctx->caller()->call("p2b", binary_public);
  auto _binary_share = ArrayView<std::array<uint64_t, 2>>(binary_share);

  std::vector<conversion::BitStream> s(in.numel());
  std::vector<conversion::BitStream> r(in.numel());

  pforeach(0, in.numel(), [&](uint64_t idx) {
    s[idx].resize(61);
    for (int i = 0; i < 61; i++) {
      s[idx][i][0] = (_binary_share[idx][0] >> i) & 0x1;
      s[idx][i][1] = (_binary_share[idx][1] >> i) & 0x1;
    }
    std::copy(edabits[idx].bshares.begin(), edabits[idx].bshares.end(),
              back_inserter(r[idx]));
  });

  auto neg_r = twos_complement(ctx->caller(), r, 61);
  auto s_binary = full_adder(ctx->caller(), s, neg_r, true);

  ArrayRef out(makeType<BShrTy>(PT_U64, 61), in.numel());
  auto _out = ArrayView<std::array<uint64_t, 2>>(out);

  pforeach(0, in.numel(), [&](uint64_t idx) {
    _out[idx][0] = 0;
    _out[idx][1] = 0;
    for (int i = 0; i < 61; i++) {
      _out[idx][0] |= static_cast<uint64_t>(s_binary[idx][i][0]) << i;
      _out[idx][1] |= static_cast<uint64_t>(s_binary[idx][i][1]) << i;
    }
  });

  return out;
}

// FIXME: This implementation is not definitely safe. It firstly compute x+r in
// binary and then open it. However, it reveals the relation between x and x+r,
// because x and r are both unsigned, and the addition is in integer. The
// correct way is manage to open x+r in field, but it maybe needs comparison
// circuit.
ArrayRef B2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  auto* edabit_state = ctx->getState<EdabitState>();
  auto* comm = ctx->getState<Communicator>();
  auto* spdzwisefield_state = ctx->getState<SpdzWiseFieldState>();

  auto key = spdzwisefield_state->key();

  (void)comm;

  using Field = SpdzWiseFieldState::Field;

  auto _in = ArrayView<std::array<uint64_t, 2>>(in);
  std::vector<conversion::BitStream> s(in.numel());
  std::vector<conversion::BitStream> r(in.numel());

  auto edabits = edabit_state->gen_edabits(ctx->caller(), PT_U64, in.numel());

  pforeach(0, in.numel(), [&](uint64_t idx) {
    s[idx].resize(61);
    for (int i = 0; i < 61; i++) {
      s[idx][i][0] = (_in[idx][0] >> i) & 0x1;
      s[idx][i][1] = (_in[idx][1] >> i) & 0x1;
    }
    std::copy(edabits[idx].bshares.begin(), edabits[idx].bshares.end(),
              back_inserter(r[idx]));
  });

  // get vector [s + r]_b
  auto s_r = full_adder(ctx->caller(), s, r, true);

  // use ArrayRef to store them.
  ArrayRef bshare_s_r(makeType<BShrTy>(PT_U64, 62), in.numel());
  auto _bshare_s_r = ArrayView<std::array<uint64_t, 2>>(bshare_s_r);

  pforeach(0, in.numel(), [&](uint64_t idx) {
    _bshare_s_r[idx][0] = 0;
    _bshare_s_r[idx][1] = 0;

    for (int i = 0; i < 62; i++) {
      _bshare_s_r[idx][0] += static_cast<uint64_t>(s_r[idx][i][0]) << i;
      _bshare_s_r[idx][1] += static_cast<uint64_t>(s_r[idx][i][1]) << i;
    }
  });

  // get share of r and mac_key
  ArrayRef ashare_r(makeType<aby3::AShrTy>(FM64), in.numel());
  ArrayRef key_share(makeType<aby3::AShrTy>(FM64), in.numel());

  auto _ashare_r = ArrayView<std::array<uint64_t, 2>>(ashare_r);
  auto _key_share = ArrayView<std::array<uint64_t, 2>>(key_share);

  pforeach(0, in.numel(), [&](uint64_t idx) {
    _ashare_r[idx] = edabits[idx].ashare;
    _key_share[idx] = key;
  });

  // get [mac]
  ArrayRef mac = ctx->caller()->call("mul_aa_sh", ashare_r, key_share);
  auto _mac = ArrayView<std::array<uint64_t, 2>>(mac);

  // open s + r
  ArrayRef opened_s_r = ctx->caller()->call("b2p", bshare_s_r);
  ArrayRef ashare_s = ctx->caller()->call("p2a", opened_s_r);

  // s = (s + r) - r
  auto _ashare_s = ArrayView<std::array<uint64_t, 4>>(ashare_s);

  pforeach(0, in.numel(), [&](uint64_t idx) {
    _ashare_s[idx][0] = Field::sub(_ashare_s[idx][0], edabits[idx].ashare[0]);
    _ashare_s[idx][1] = Field::sub(_ashare_s[idx][1], edabits[idx].ashare[1]);
    _ashare_s[idx][2] = Field::sub(_ashare_s[idx][2], _mac[idx][0]);
    _ashare_s[idx][3] = Field::sub(_ashare_s[idx][3], _mac[idx][1]);
  });

  return ashare_s;
}

// Reference:
// 5.4.1 Semi-honest Security
// https://eprint.iacr.org/2018/403.pdf
//
// Latency: 2.
//
// Aby3 paper algorithm reference.
//
// P1 & P3 locally samples c1.
// P2 & P3 locally samples c3.
//
// P3 (the OT sender) defines two messages.
//   m{i} := (i^b1^b3)−c1−c3 for i in {0, 1}
// P2 (the receiver) defines his input to be b2 in order to learn the message
//   c2 = m{b2} = (b2^b1^b3)−c1−c3 = b − c1 − c3.
// P1 (the helper) also knows b2 and therefore the three party OT can be used.
//
// However, to make this a valid 2-out-of-3 secret sharing, P1 needs to learn
// c2.
//
// Current implementation
// - P2 could send c2 resulting in 2 rounds and 4k bits of communication.
//
// TODO:
// - Alternatively, the three-party OT procedure can be repeated (in parallel)
// with again party 3 playing the sender with inputs m0,mi so that party 1
// (the receiver) with input bit b2 learns the message c2 (not m[b2]) in the
// first round, totaling 6k bits and 1 round.
ArrayRef BitInject::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_TRACE_MPC_LEAF(ctx, in);

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto* in_ty = in.eltype().as<BShrTy>();
  const size_t in_nbits = in_ty->nbits();

  SPU_ENFORCE(in_nbits == 1,
              "invalid nbits={}, bit inject only supports single bit",
              in_nbits);

  ArrayRef out(makeType<aby3::AShrTy>(field), in.numel());

  using Field = SpdzWiseFieldState::Field;

  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  // P0 as the helper/dealer, helps to prepare correlated randomness.
  // P1, P2 as the receiver and sender of OT.
  size_t pivot;
  prg_state->fillPubl(absl::MakeSpan(&pivot, 1));
  size_t P0 = pivot % 3;        // helper
  size_t P1 = (pivot + 1) % 3;  // receiver
  size_t P2 = (pivot + 2) % 3;  // sender

  DISPATCH_UINT_PT_TYPES(in_ty->getBacktype(), "_", [&]() {
    using BShrT = ScalarT;

    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      using AShrT = ring2k_t;

      auto _in = ArrayView<std::array<BShrT, 2>>(in);
      auto _out = ArrayView<std::array<AShrT, 2>>(out);

      const size_t total_nbits = in.numel() * in_nbits;
      std::vector<AShrT> r0(total_nbits);
      std::vector<AShrT> r1(total_nbits);
      prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

      pforeach(0, total_nbits, [&](uint64_t idx) {
        r0[idx] = Field::modp(r0[idx]);
        r1[idx] = Field::modp(r1[idx]);
      });

      if (comm->getRank() == P0) {
        // the helper
        auto b2 = bitDecompose(ArrayView<BShrT>(getShare(in, 1)), in_nbits);

        // gen masks with helper.
        std::vector<AShrT> m0(total_nbits);
        std::vector<AShrT> m1(total_nbits);
        prg_state->fillPrssPair(absl::MakeSpan(m0), {}, false, true);
        prg_state->fillPrssPair(absl::MakeSpan(m1), {}, false, true);

        // build selected mask
        SPU_ENFORCE(b2.size() == m0.size() && b2.size() == m1.size());
        pforeach(0, total_nbits,
                 [&](int64_t idx) { m0[idx] = !b2[idx] ? m0[idx] : m1[idx]; });

        // send selected masked to receiver.
        comm->sendAsync<AShrT>(P1, m0, "mc");

        auto c1 = bitCompose<AShrT>(r0, in_nbits);
        auto c2 = comm->recv<AShrT>(P1, "c2");

        pforeach(0, in.numel(), [&](int64_t idx) {
          _out[idx][0] = Field::modp(c1[idx]);
          _out[idx][1] = Field::modp(c2[idx]);
        });
      } else if (comm->getRank() == P1) {
        // the receiver
        prg_state->fillPrssPair(absl::MakeSpan(r0), {}, false, false);
        prg_state->fillPrssPair(absl::MakeSpan(r0), {}, false, false);

        auto b2 = bitDecompose(ArrayView<BShrT>(getShare(in, 0)), in_nbits);

        // ot.recv
        auto mc = comm->recv<AShrT>(P0, "mc");
        auto m0 = comm->recv<AShrT>(P2, "m0");
        auto m1 = comm->recv<AShrT>(P2, "m1");

        // rebuild c2 = (b1^b2^b3)-c1-c3
        pforeach(0, total_nbits, [&](int64_t idx) {
          mc[idx] = !b2[idx] ? m0[idx] ^ mc[idx] : m1[idx] ^ mc[idx];
        });
        auto c2 = bitCompose<AShrT>(mc, in_nbits);
        comm->sendAsync<AShrT>(P0, c2, "c2");
        auto c3 = bitCompose<AShrT>(r1, in_nbits);

        pforeach(0, in.numel(), [&](int64_t idx) {
          _out[idx][0] = Field::modp(c2[idx]);
          _out[idx][1] = Field::modp(c3[idx]);
        });
      } else if (comm->getRank() == P2) {
        // the sender.
        auto c3 = bitCompose<AShrT>(r0, in_nbits);
        auto c1 = bitCompose<AShrT>(r1, in_nbits);

        // c3 = r0, c1 = r1
        // let mi := (i^b1^b3)−c1−c3 for i in {0, 1}
        // reuse r's memory for m
        pforeach(0, in.numel(), [&](int64_t idx) {
          auto xx = _in[idx][0] ^ _in[idx][1];
          for (size_t bit = 0; bit < in_nbits; bit++) {
            size_t flat_idx = idx * in_nbits + bit;
            AShrT t = Field::add(r0[flat_idx], r1[flat_idx]);
            r0[flat_idx] = Field::sub(((xx >> bit) & 0x1), t);
            r1[flat_idx] = Field::sub(((~xx >> bit) & 0x1), t);
          }
        });

        // gen masks with helper.
        std::vector<AShrT> m0(total_nbits);
        std::vector<AShrT> m1(total_nbits);
        prg_state->fillPrssPair({}, absl::MakeSpan(m0), true, false);
        prg_state->fillPrssPair({}, absl::MakeSpan(m1), true, false);
        pforeach(0, total_nbits, [&](int64_t idx) {
          m0[idx] ^= r0[idx];
          m1[idx] ^= r1[idx];
        });

        comm->sendAsync<AShrT>(P1, m0, "m0");
        comm->sendAsync<AShrT>(P1, m1, "m1");

        pforeach(0, in.numel(), [&](int64_t idx) {
          _out[idx][0] = Field::modp(c3[idx]);
          _out[idx][1] = Field::modp(c1[idx]);
        });
      } else {
        SPU_THROW("expected party=3, got={}", comm->getRank());
      }
    });
  });

  return out;
}

}  // namespace spu::mpc::spdzwisefield