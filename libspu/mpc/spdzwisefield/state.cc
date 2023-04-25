
#include "libspu/mpc/spdzwisefield/state.h"

// #include <fstream>
#include <random>

#include "libspu/core/array_ref.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/spdzwisefield/type.h"

#define MYLOG(x) \
  if (comm->getRank() == 0) std::cout << x << std::endl

namespace spu::mpc {

ArrayRef BeaverState::gen_bin_triples(Object* ctx, PtType out_type, size_t size,
                                      size_t batch_size, size_t bucket_size) {
  auto* comm = ctx->getState<Communicator>();
  auto* prg = ctx->getState<PrgState>();

  int compress =
      sizeof(typename beaver::BTDataType) / SizeOf(PtTypeToField(out_type));
  int need = (size - 1) / compress + 1;
  int64_t triples_needed = need - trusted_triples_bin_->size();

  if (triples_needed > 0) {
    size_t num_per_batch = batch_size * bucket_size + bucket_size;
    size_t num_batches = (triples_needed - 1) / batch_size + 1;

    std::vector<beaver::BinaryTriple> new_triples(num_per_batch * num_batches);

    std::vector<beaver::BTDataType> r0(num_per_batch * num_batches);
    std::vector<beaver::BTDataType> r1(num_per_batch * num_batches);

    std::array<std::vector<beaver::BTDataType>, 2> a, b, c;

    a[0].resize(num_per_batch * num_batches);
    a[1].resize(num_per_batch * num_batches);
    b[0].resize(num_per_batch * num_batches);
    b[1].resize(num_per_batch * num_batches);
    c[0].resize(num_per_batch * num_batches);
    c[1].resize(num_per_batch * num_batches);

    prg->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));
    prg->fillPrssPair(absl::MakeSpan(a[0]), absl::MakeSpan(a[1]));
    prg->fillPrssPair(absl::MakeSpan(b[0]), absl::MakeSpan(b[1]));

    pforeach(0, num_per_batch * num_batches, [&](int64_t idx) {
      r0[idx] = (a[0][idx] & b[0][idx]) ^  //
                (a[0][idx] & b[1][idx]) ^  //
                (a[1][idx] & b[0][idx]) ^  //
                (r0[idx] ^ r1[idx]);
    });

    r1 = comm->rotate<beaver::BTDataType>(r0, "cacbin");  // comm => 1, k

    pforeach(0, num_per_batch * num_batches, [&](int64_t idx) {
      new_triples[idx][0] = {a[0][idx], a[1][idx]};
      new_triples[idx][1] = {b[0][idx], b[1][idx]};
      new_triples[idx][2] = {r0[idx], r1[idx]};
    });

    for (size_t i = 0; i < num_batches; i++) {
      auto trusted_triples =
          cut_and_choose(ctx, new_triples.begin() + i * num_per_batch,
                         batch_size, bucket_size, bucket_size);

      trusted_triples_bin_->insert(trusted_triples_bin_->end(),
                                   trusted_triples.begin(),
                                   trusted_triples.end());
    }
  }

  return DISPATCH_UINT_PT_TYPES(out_type, "_", [&]() {
    using OutT = ScalarT;

    size_t size_bit = sizeof(OutT) * 8;
    OutT mask = (1 << size_bit) - 1;

    ArrayRef out(makeType<spdzwisefield::BinTripleTy>(out_type), size);
    auto out_view = ArrayView<std::array<std::array<OutT, 2>, 3>>(out);

    pforeach(0, size, [&](uint64_t idx) {
      int row = idx / compress;
      int col = idx % compress;
      auto& [a, b, c] = out_view[idx];
      const auto [x, y, z] = trusted_triples_bin_->at(row);

      a[0] = (x[0] >> (size_bit * col)) & mask;
      a[1] = (x[1] >> (size_bit * col)) & mask;
      b[0] = (y[0] >> (size_bit * col)) & mask;
      b[1] = (y[1] >> (size_bit * col)) & mask;
      c[0] = (z[0] >> (size_bit * col)) & mask;
      c[1] = (z[1] >> (size_bit * col)) & mask;
    });

    trusted_triples_bin_->erase(trusted_triples_bin_->begin(),
                                trusted_triples_bin_->begin() + need);

    return out;
  });
}

// Refer to:
// 2.10 Triple Verification Using Another (Without Opening)
// FLNW17 (High-Throughput Secure Three-Party Computation for Malicious
// Adversaries and an Honest Majority)

std::vector<beaver::BinaryTriple> BeaverState::cut_and_choose(
    Object* ctx, typename std::vector<beaver::BinaryTriple>::iterator data,
    size_t batch_size, size_t bucket_size, size_t C) {
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  // generate random seed for shuffle

  std::vector<uint64_t> r0(1);
  std::vector<uint64_t> r1(1);

  prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));
  auto r2 = comm->rotate<uint64_t>(r1, "cacbin");

  uint64_t seed = r0[0] + r1[0] + r2[0];

  (void)seed;

  size_t num_per_batch = batch_size * bucket_size + C;
  std::mt19937 rng(seed);
  std::shuffle(data, data + num_per_batch, rng);

  // open first C triples

  std::vector<beaver::BTDataType> send_buffer(C * 3);

  pforeach(0, C, [&](uint64_t idx) {
    send_buffer[idx * 3] = data[idx][0][1];
    send_buffer[idx * 3 + 1] = data[idx][1][1];
    send_buffer[idx * 3 + 2] = data[idx][2][1];
  });

  auto recv_buffer = comm->rotate<beaver::BTDataType>(send_buffer, "cacbin");

  pforeach(0, C, [&](uint64_t idx) {
    beaver::BTDataType a =
        recv_buffer[idx * 3] ^ data[idx][0][0] ^ data[idx][0][1];
    beaver::BTDataType b =
        recv_buffer[idx * 3 + 1] ^ data[idx][1][0] ^ data[idx][1][1];
    beaver::BTDataType c =
        recv_buffer[idx * 3 + 2] ^ data[idx][2][0] ^ data[idx][2][1];

    if ((a & b) != c) {
      throw std::runtime_error(
          "Verification when check first C triples in Cut and Choose failed at "
          "index " +
          std::to_string(idx));
    }
  });

  data += C;

  // PROTOCOL 2.24 (Triple Verif. Using Another Without Opening)

  send_buffer.resize(batch_size * (bucket_size - 1) * 2);

  pforeach(0, batch_size, [&](uint64_t idx) {
    for (size_t i = 0; i < bucket_size - 1; i++) {
      send_buffer[idx * (bucket_size - 1) * 2 + i * 2] =
          data[idx * bucket_size][0][1] ^ data[idx * bucket_size + i + 1][0][1];
      send_buffer[idx * (bucket_size - 1) * 2 + i * 2 + 1] =
          data[idx * bucket_size][1][1] ^ data[idx * bucket_size + i + 1][1][1];
    }
  });

  recv_buffer.resize(batch_size * (bucket_size - 1) * 2);

  recv_buffer = comm->rotate<beaver::BTDataType>(send_buffer, "cacbin");

  std::vector<beaver::BTDataType> _opened2(batch_size * (bucket_size - 1) * 2);

  pforeach(0, batch_size, [&](uint64_t idx) {
    for (size_t i = 0; i < bucket_size - 1; i++) {
      _opened2[idx * (bucket_size - 1) * 2 + i * 2] =
          (data[idx * bucket_size][0][0] ^
           data[idx * bucket_size + i + 1][0][0]) ^
          (data[idx * bucket_size][0][1] ^
           data[idx * bucket_size + i + 1][0][1]) ^
          recv_buffer[idx * (bucket_size - 1) * 2 + i * 2];

      _opened2[idx * (bucket_size - 1) * 2 + i * 2 + 1] =
          (data[idx * bucket_size][1][0] ^
           data[idx * bucket_size + i + 1][1][0]) ^
          (data[idx * bucket_size][1][1] ^
           data[idx * bucket_size + i + 1][1][1]) ^
          recv_buffer[idx * (bucket_size - 1) * 2 + i * 2 + 1];
    }
  });

  std::vector<std::array<beaver::BTDataType, 2>> z(batch_size *
                                                   (bucket_size - 1));

  send_buffer.resize(batch_size * (bucket_size - 1));

  pforeach(0, batch_size, [&](uint64_t idx) {
    beaver::BinaryTriple trusted = data[idx * bucket_size];
    for (size_t i = 0; i < bucket_size - 1; i++) {
      auto rho = _opened2[idx * (bucket_size - 1) * 2 + i * 2];
      auto sigma = _opened2[idx * (bucket_size - 1) * 2 + i * 2 + 1];

      beaver::BinaryTriple untrusted = data[idx * bucket_size + i + 1];

      auto x1 = trusted[2][0] ^ untrusted[2][0] ^ (sigma & untrusted[0][0]) ^
                (rho & untrusted[1][0]);
      auto x2 = trusted[2][1] ^ untrusted[2][1] ^ (sigma & untrusted[0][1]) ^
                (rho & untrusted[1][1]);

      if (comm->getRank() == 0) {
        x1 ^= (rho & sigma);
      } else if (comm->getRank() == 2) {
        x2 ^= (rho & sigma);
      }

      z[idx * (bucket_size - 1) + i][0] = x1;
      z[idx * (bucket_size - 1) + i][1] = x2;
      send_buffer[idx * (bucket_size - 1) + i] = x2;
    }
  });

  recv_buffer.resize(batch_size * (bucket_size - 1));
  recv_buffer = comm->rotate<beaver::BTDataType>(send_buffer, "cacbin");

  pforeach(0, batch_size * (bucket_size - 1), [&](uint64_t idx) {
    assert((z[idx][0] ^ z[idx][1] ^ recv_buffer[idx]) == 0);
  });

  std::vector<beaver::BinaryTriple> res(batch_size);

  pforeach(0, batch_size,
           [&](uint64_t idx) { res[idx] = data[idx * bucket_size]; });

  return res;
}

// ArrayRef EdabitState::gen_edabits(Object* ctx, PtType out_type, size_t size,
//                                   size_t batch_size, size_t bucket_size) {
//   SPU_ENFORCE(out_type == PT_U64, "Edabit only supports u64 output type");
//   return ArrayRef();
//   // auto* comm = ctx->getState<Communicator>();
// }

// std::vector<conversion::Edabit> EdabitState::cut_and_choose(
//     Object* ctx, typename std::vector<conversion::Edabit>::iterator data,
//     size_t batch_size, size_t bucket_size, size_t C) {
//   return std::vector<conversion::Edabit>();
// }

std::vector<BitStream> full_adder_with_check(
    Object* ctx, const std::vector<BitStream>& lhs,
    const std::vector<BitStream>& rhs) {
  SPU_ENFORCE(lhs.size() == rhs.size(), "lhs and rhs must have same size");

  size_t size = (lhs.size() - 1) / 64 + 1;
  size_t nbits = lhs[0].size();

  std::vector<std::array<bool, 2>> c(lhs.size(), {false, false});
  std::vector<BitStream> result(lhs.size() + 1);

  for (uint64_t i = 0; i < nbits; i++) {
    ArrayRef inner_lhs(makeType<spdzwisefield::BShrTy>(PT_U64, 64), size);
    ArrayRef inner_rhs(makeType<spdzwisefield::BShrTy>(PT_U64, 64), size);

    auto _inner_lhs = ArrayView<std::array<uint64_t, 2>>(inner_lhs);
    auto _inner_rhs = ArrayView<std::array<uint64_t, 2>>(inner_rhs);

    pforeach(0, size, [&](uint64_t idx) {
      auto& lhs_val = _inner_lhs[idx];
      auto& rhs_val = _inner_rhs[idx];

      for (uint64_t j = 0; j < 64; j++) {
        lhs_val[0] |= (lhs[idx * 64 + j][i][0] ^ c[idx * 64 + j][0]) << j;
        lhs_val[1] |= (lhs[idx * 64 + j][i][1] ^ c[idx * 64 + j][1]) << j;

        rhs_val[0] |= (rhs[idx * 64 + j][i][0] ^ c[idx * 64 + j][0]) << j;
        rhs_val[1] |= (rhs[idx * 64 + j][i][1] ^ c[idx * 64 + j][1]) << j;

        result[idx * 64 + j][i][0] = lhs[idx * 64 + j][i][0] ^
                                     rhs[idx * 64 + j][i][0] ^
                                     c[idx * 64 + j][0];

        result[idx * 64 + j][i][1] = lhs[idx * 64 + j][i][1] ^
                                     rhs[idx * 64 + j][i][1] ^
                                     c[idx * 64 + j][1];
      }
    });

    auto res = ctx->call("and_bb", inner_lhs, inner_rhs);

    pforeach(0, size, [&](uint64_t idx) {
      auto& res_val = _inner_lhs[idx];

      for (uint64_t j = 0; j < 64; j++) {
        c[idx * 64 + j][0] = (res_val[0] >> j) & 1;
        c[idx * 64 + j][1] = (res_val[1] >> j) & 1;
      }
    });
  }

  pforeach(0, lhs.size(), [&](uint64_t idx) {
    result[idx][nbits][0] = c[idx][0];
    result[idx][nbits][1] = c[idx][1];
  });

  return result;
}

}  // namespace spu::mpc