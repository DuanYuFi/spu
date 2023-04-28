
#include "libspu/mpc/spdzwisefield/state.h"

// #include <fstream>
#include <random>

#include "libspu/core/array_ref.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/spdzwisefield/type.h"
#include "libspu/mpc/spdzwisefield/value.h"

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
    auto hash_algo = std::make_shared<yacl::crypto::Blake3Hash>();
    auto hash_algo2 = std::make_shared<yacl::crypto::Blake3Hash>();

    size_t num_per_batch = batch_size * bucket_size + bucket_size;
    size_t num_batches = (triples_needed - 1) / batch_size + 1;

    std::vector<beaver::BinaryTriple> new_triples(num_per_batch * num_batches);

    std::vector<beaver::BTDataType> r0(num_per_batch * num_batches);
    std::vector<beaver::BTDataType> r1(num_per_batch * num_batches);

    std::array<std::vector<beaver::BTDataType>, 2> a;
    std::array<std::vector<beaver::BTDataType>, 2> b;
    std::array<std::vector<beaver::BTDataType>, 2> c;

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

    r1 = comm->rotate<beaver::BTDataType>(r0, "cac.bin");  // comm => 1, k

    pforeach(0, num_per_batch * num_batches, [&](int64_t idx) {
      new_triples[idx][0] = {a[0][idx], a[1][idx]};
      new_triples[idx][1] = {b[0][idx], b[1][idx]};
      new_triples[idx][2] = {r0[idx], r1[idx]};
    });

    for (size_t i = 0; i < num_batches; i++) {
      auto trusted_triples = cut_and_choose(
          ctx, new_triples.begin() + i * num_per_batch, hash_algo, hash_algo2,
          batch_size, bucket_size, bucket_size);

      trusted_triples_bin_->insert(trusted_triples_bin_->end(),
                                   trusted_triples.begin(),
                                   trusted_triples.end());
    }

    std::vector<uint8_t> hash = hash_algo->CumulativeHash();
    std::vector<uint8_t> hash2 = hash_algo2->CumulativeHash();

    std::string hash_str2(reinterpret_cast<char*>(hash2.data()), 32);

    auto res = comm->rotate<uint8_t>(hash, "cac.bin");

    std::string hash_str(reinterpret_cast<char*>(res.data()), 32);

    // SPU_ENFORCE(hash_str == hash_str2, "hash mismatch");
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
    Object* ctx, std::vector<beaver::BinaryTriple>::iterator data,
    const std::shared_ptr<yacl::crypto::Blake3Hash>& hash_algo,
    const std::shared_ptr<yacl::crypto::Blake3Hash>& hash_algo2,
    size_t batch_size, size_t bucket_size, size_t C) {
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  // generate random seed for shuffle

  std::vector<uint64_t> r0(1);
  std::vector<uint64_t> r1(1);

  prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));
  auto r2 = comm->rotate<uint64_t>(r1, "cac.bin");

  uint64_t seed = r0[0] + r1[0] + r2[0];

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

  auto recv_buffer = comm->rotate<beaver::BTDataType>(send_buffer, "cac.bin");

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

  recv_buffer = comm->rotate<beaver::BTDataType>(send_buffer, "cac.bin");

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

  // send_buffer.resize(batch_size * (bucket_size - 1));

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
      // send_buffer[idx * (bucket_size - 1) + i] = x2;
    }
  });

  for (uint64_t idx = 0; idx < batch_size; idx++) {
    for (uint64_t i = 0; i < bucket_size - 1; i++) {
      auto x1 = z[idx * (bucket_size - 1) + i][0];
      auto x2 = z[idx * (bucket_size - 1) + i][1];

      hash_algo->Update(std::to_string(static_cast<uint64_t>(
                            (x1 ^ x2) & 0xFFFFFFFFFFFFFFFF)) +
                        std::to_string(static_cast<uint64_t>((x1 ^ x2) >> 64)));

      hash_algo2->Update(
          std::to_string(static_cast<uint64_t>(x2 & 0xFFFFFFFFFFFFFFFF)) +
          std::to_string(static_cast<uint64_t>(x2 >> 64)));
    }
  }

  // recv_buffer.resize(batch_size * (bucket_size - 1));
  // recv_buffer = comm->rotate<beaver::BTDataType>(send_buffer, "cac.bin");

  // pforeach(0, batch_size * (bucket_size - 1), [&](uint64_t idx) {
  //   assert((z[idx][0] ^ z[idx][1] ^ recv_buffer[idx]) == 0);
  // });

  std::vector<beaver::BinaryTriple> res(batch_size);

  pforeach(0, batch_size,
           [&](uint64_t idx) { res[idx] = data[idx * bucket_size]; });

  return res;
}

/*

    ================================== Edabit ==================================

*/

std::vector<conversion::Edabit> EdabitState::gen_edabits(Object* ctx,
                                                         PtType arith_type,
                                                         size_t size,
                                                         size_t batch_size,
                                                         size_t bucket_size) {
  SPU_ENFORCE(arith_type == PT_U64, "Edabit only supports u64 arithmetic type");

  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  using Field = SpdzWiseFieldState::Field;

  if (trusted_edabits_->size() < size) {
    size_t batches = (size - trusted_edabits_->size() - 1) / batch_size + 1;
    size_t size_per_batch = batch_size * bucket_size + bucket_size;
    size_t total = batches * size_per_batch;

    std::vector<conversion::Edabit> myself(total), prev(total), next(total);
    std::vector<uint64_t> randbits(total), share_prev(total), share_next(total);
    std::vector<uint64_t> arith_share_prev(total), arith_share_next(total);

    prg_state->fillPrssPair(absl::MakeSpan(share_prev),
                            absl::MakeSpan(share_next));
    prg_state->fillPrssPair(absl::MakeSpan(arith_share_prev),
                            absl::MakeSpan(arith_share_next));
    prg_state->fillPriv(absl::MakeSpan(randbits));

    pforeach(0, total, [&](uint64_t idx) {
      share_prev[idx] = Field::modp(share_prev[idx]);
      share_next[idx] = Field::modp(share_next[idx]);
      arith_share_prev[idx] = Field::modp(arith_share_prev[idx]);
      arith_share_next[idx] = Field::modp(arith_share_next[idx]);
      randbits[idx] = Field::modp(randbits[idx]);
    });

    // MYLOG("After generate randbits");
    // sleep(comm->getRank());
    // std::cout << "random = " << randbits[0] << std::endl;

    std::vector<uint64_t> buffer(total * 2);

    pforeach(0, total, [&](uint64_t idx) {
      auto& edabit = myself[idx];
      auto& prev_edabit = prev[idx];

      edabit.ashare[1] = arith_share_next[idx];
      edabit.ashare[0] = Field::sub(randbits[idx], edabit.ashare[1]);

      buffer[idx * 2] = edabit.ashare[0];

      edabit.ashare[0] = Field::add(edabit.ashare[0], arith_share_prev[idx]);

      for (uint64_t i = 0; i < nbits_; i++) {
        edabit.bshares[i][1] = ((share_next[idx] & 1) != 0U);
        edabit.bshares[i][0] =
            ((edabit.bshares[i][1] ^ (randbits[idx] & 1)) != 0U);

        buffer[idx * 2 + 1] |= static_cast<uint64_t>(edabit.bshares[i][0]) << i;

        prev_edabit.bshares[i][0] = share_prev[idx] & 1;
        prev_edabit.bshares[i][1] = 0;

        share_prev[idx] >>= 1;
        randbits[idx] >>= 1;
        share_next[idx] >>= 1;
      }
    });

    // MYLOG("Before rotate");

    auto recv_buffer = comm->rotate<uint64_t>(buffer, "cac.edabit");

    pforeach(0, total, [&](uint64_t idx) {
      auto& edabit = myself[idx];
      auto& next_edabit = next[idx];

      edabit.ashare[1] = Field::add(edabit.ashare[1], recv_buffer[idx * 2]);

      for (uint64_t i = 0; i < nbits_; i++) {
        next_edabit.bshares[i][1] = recv_buffer[idx * 2 + 1] & 1;
        next_edabit.bshares[i][0] = 0;
        recv_buffer[idx * 2 + 1] >>= 1;
      }
    });

    // std::vector<conversion::Edabit> test(3);
    // test[0].ashare = myself[0].ashare;

    // switch (comm->getRank()) {
    //   case 0: {
    //     std::copy_n(myself[0].bshares.begin(), nbits_,
    //     test[0].bshares.begin()); std::copy_n(next[0].bshares.begin(),
    //     nbits_, test[1].bshares.begin());
    //     std::copy_n(prev[0].bshares.begin(), nbits_,
    //     test[2].bshares.begin()); break;
    //   }
    //   case 1: {
    //     std::copy_n(prev[0].bshares.begin(), nbits_,
    //     test[0].bshares.begin()); std::copy_n(myself[0].bshares.begin(),
    //     nbits_, test[1].bshares.begin());
    //     std::copy_n(next[0].bshares.begin(), nbits_,
    //     test[2].bshares.begin()); break;
    //   }
    //   case 2: {
    //     std::copy_n(next[0].bshares.begin(), nbits_,
    //     test[0].bshares.begin()); std::copy_n(prev[0].bshares.begin(),
    //     nbits_, test[1].bshares.begin());
    //     std::copy_n(myself[0].bshares.begin(), nbits_,
    //     test[2].bshares.begin()); break;
    //   }

    //   default:
    //     break;
    // }

    // auto open = open_edabits(ctx, test.begin(), 3);

    // if (comm->getRank() == 0) {
    //   open[0].print();
    //   open[1].print();
    //   open[2].print();
    // }

    std::vector<conversion::BitStream> b0, b1, b2;

    // MYLOG("Before prepare data");

    for (uint64_t idx = 0; idx < total; idx++) {
      switch (comm->getRank()) {
        case 0: {
          b0.emplace_back(myself[idx].bshares.begin(),
                          myself[idx].bshares.end());
          b1.emplace_back(next[idx].bshares.begin(), next[idx].bshares.end());
          b2.emplace_back(prev[idx].bshares.begin(), prev[idx].bshares.end());
          break;
        }
        case 1: {
          b0.emplace_back(prev[idx].bshares.begin(), prev[idx].bshares.end());
          b1.emplace_back(myself[idx].bshares.begin(),
                          myself[idx].bshares.end());
          b2.emplace_back(next[idx].bshares.begin(), next[idx].bshares.end());
          break;
        }
        case 2: {
          b0.emplace_back(next[idx].bshares.begin(), next[idx].bshares.end());
          b1.emplace_back(prev[idx].bshares.begin(), prev[idx].bshares.end());
          b2.emplace_back(myself[idx].bshares.begin(),
                          myself[idx].bshares.end());
          break;
        }

        default:
          break;
      }
    }

    // MYLOG("Before full_adder");

    // std::vector<conversion::Edabit> test2(2);
    // test2[0].ashare = myself[0].ashare;
    // test2[1].ashare = myself[1].ashare;

    // std::copy_n(b0[0].begin(), nbits_, test2[0].bshares.begin());
    // std::copy_n(b1[0].begin(), nbits_, test2[1].bshares.begin());

    // auto open2 = open_edabits(ctx, test2.begin(), 2);

    // sleep(comm->getRank());
    // open2[0].print(true);
    // open2[1].print(true);

    auto middle = full_adder(ctx, b0, b1, false);

    // auto open3 = open_bits(ctx, middle[0]);
    // if (comm->getRank() == 0) {
    //   for (auto each : open3) {
    //     std::cout << each;
    //   }
    //   std::cout << std::endl;
    // }

    auto result = full_adder(ctx, middle, b2, false);

    ArrayRef redundant_bits(makeType<spdzwisefield::BShrTy>(PT_U8, 1),
                            total * 3);
    auto _redundant_bits = ArrayView<std::array<uint8_t, 2>>(redundant_bits);
    pforeach(0, total, [&](uint64_t idx) {
      _redundant_bits[idx * 3][0] = result[idx][nbits_ - 1][0];
      _redundant_bits[idx * 3][1] = result[idx][nbits_ - 1][1];

      _redundant_bits[idx * 3 + 1][0] = result[idx][nbits_][0];
      _redundant_bits[idx * 3 + 1][1] = result[idx][nbits_][1];

      _redundant_bits[idx * 3 + 2][0] = result[idx][nbits_ + 1][0];
      _redundant_bits[idx * 3 + 2][1] = result[idx][nbits_ + 1][1];
    });

    // MYLOG("Before bitinject");

    ArrayRef redundant_arith = ctx->call("bitinject", redundant_bits);
    auto _redundant_arith = ArrayView<std::array<uint64_t, 2>>(redundant_arith);

    pforeach(0, total, [&](uint64_t idx) {
      auto& edabit = myself[idx];
      std::copy_n(result[idx].begin(), nbits_ - 1, edabit.bshares.begin());
      edabit.bshares[nbits_ - 1] = {false, false};

      std::array<uint64_t, 2> substraction;
      substraction[0] =
          Field::add(_redundant_arith[idx * 3][0],
                     Field::mul(_redundant_arith[idx * 3 + 1][0], 2),
                     Field::mul(_redundant_arith[idx * 3 + 2][0], 4));

      substraction[1] =
          Field::add(_redundant_arith[idx * 3][1],
                     Field::mul(_redundant_arith[idx * 3 + 1][1], 2),
                     Field::mul(_redundant_arith[idx * 3 + 2][1], 4));

      edabit.ashare[0] =
          Field::sub(edabit.ashare[0], Field::mul(substraction[0], 1ULL << 60));
      edabit.ashare[1] =
          Field::sub(edabit.ashare[1], Field::mul(substraction[1], 1ULL << 60));
    });

    for (uint64_t idx = 0; idx < batches; idx++) {
      auto trusted_edabits =
          cut_and_choose(ctx, myself.begin() + idx * size_per_batch, batch_size,
                         bucket_size, bucket_size);

      if (trusted_edabits.size() == 0) {
        return std::vector<conversion::Edabit>();
      }

      trusted_edabits_->insert(trusted_edabits_->end(), trusted_edabits.begin(),
                               trusted_edabits.end());
    }
  }

  std::vector<conversion::Edabit> ret(size);
  std::copy_n(trusted_edabits_->begin(), size, ret.begin());
  trusted_edabits_->erase(trusted_edabits_->begin(),
                          trusted_edabits_->begin() + size);

  return ret;
}

std::vector<conversion::Edabit> EdabitState::cut_and_choose(
    Object* ctx, typename std::vector<conversion::Edabit>::iterator data,
    size_t batch_size, size_t bucket_size, size_t C) {
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();
  // generate random seed for shuffle

  // MYLOG("In cut and choose");

  std::vector<uint64_t> r0(1);
  std::vector<uint64_t> r1(1);

  prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));
  auto r2 = comm->rotate<uint64_t>(r1, "cac.edabit");

  uint64_t seed = r0[0] + r1[0] + r2[0];
  size_t size_per_batch = batch_size * bucket_size + C;

  std::mt19937 rng(seed);
  std::shuffle(data, data + size_per_batch, rng);

  std::vector<conversion::PubEdabit> opened = open_edabits(ctx, data, C);

  check_edabits(opened);

  return std::vector<conversion::Edabit>();
}

/*
  ==================================================================
  ==             Circuits for generating Edabit                   ==
  ==================================================================
*/

std::vector<conversion::PubEdabit> open_edabits(
    Object* ctx, typename std::vector<conversion::Edabit>::iterator edabits,
    size_t n) {
  auto* comm = ctx->getState<Communicator>();

  using Field = SpdzWiseFieldState::Field;

  std::vector<conversion::PubEdabit> ret(n);
  std::vector<uint64_t> buffer(n * 2);

  pforeach(0, n, [&](uint64_t idx) {
    const auto& edabit = edabits[idx];
    buffer[idx * 2] = edabit.ashare[1];

    for (size_t i = 0; i < EdabitState::nbits_; i++) {
      buffer[idx * 2 + 1] |= static_cast<uint64_t>(edabit.bshares[i][1]) << i;
    }
  });

  auto recv = comm->rotate<uint64_t>(buffer, "cac.edabit");

  pforeach(0, n, [&](uint64_t idx) {
    const auto& edabit = edabits[idx];
    ret[idx].dataA =
        Field::add(recv[idx * 2], edabit.ashare[0], edabit.ashare[1]);

    for (size_t i = 0; i < EdabitState::nbits_; i++) {
      ret[idx].dataB[i] = (((recv[idx * 2 + 1] >> i) & 1) != 0U);
      ret[idx].dataB[i] =
          ret[idx].dataB[i] ^ edabit.bshares[i][0] ^ edabit.bshares[i][1];
    }
  });

  return ret;
}

ArrayRef semi_honest_and_bb(Object* ctx, const ArrayRef& lhs,
                            const ArrayRef& rhs) {
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  const auto* lhs_ty = lhs.eltype().as<spdzwisefield::BShrTy>();
  const auto* rhs_ty = rhs.eltype().as<spdzwisefield::BShrTy>();

  const size_t out_nbits = std::max(lhs_ty->nbits(), rhs_ty->nbits());
  const PtType out_btype = spdzwisefield::calcBShareBacktype(out_nbits);

  ArrayRef out(makeType<spdzwisefield::BShrTy>(out_btype, out_nbits),
               lhs.numel());

  return DISPATCH_UINT_PT_TYPES(rhs_ty->getBacktype(), "_", [&]() {
    using RhsT = ScalarT;
    auto _rhs = ArrayView<std::array<RhsT, 2>>(rhs);

    return DISPATCH_UINT_PT_TYPES(lhs_ty->getBacktype(), "_", [&]() {
      using LhsT = ScalarT;
      auto _lhs = ArrayView<std::array<LhsT, 2>>(lhs);

      return DISPATCH_UINT_PT_TYPES(out_btype, "_", [&]() {
        using OutT = ScalarT;

        std::vector<OutT> r0(lhs.numel());
        std::vector<OutT> r1(lhs.numel());
        prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

        // z1 = (x1 & y1) ^ (x1 & y2) ^ (x2 & y1) ^ (r0 ^ r1);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          r0[idx] = (_lhs[idx][0] & _rhs[idx][0]) ^
                    (_lhs[idx][0] & _rhs[idx][1]) ^
                    (_lhs[idx][1] & _rhs[idx][0]) ^ (r0[idx] ^ r1[idx]);
        });

        r1 = comm->rotate<OutT>(r0, "andbb");  // comm => 1, k

        auto _out = ArrayView<std::array<OutT, 2>>(out);
        pforeach(0, lhs.numel(), [&](int64_t idx) {
          _out[idx][0] = r0[idx];
          _out[idx][1] = r1[idx];
        });
        return out;
      });
    });
  });
}

std::vector<conversion::BitStream> full_adder(
    Object* ctx, std::vector<conversion::BitStream> lhs,
    std::vector<conversion::BitStream> rhs, bool with_check) {
  SPU_ENFORCE(lhs.size() == rhs.size(), "lhs and rhs must have same size");

  auto* comm = ctx->getState<Communicator>();
  (void)comm;

  size_t size = (lhs.size() - 1) / 64 + 1;
  size_t nbits = std::max(lhs[0].size(), rhs[0].size());

  pforeach(0, lhs.size(), [&](uint64_t idx) {
    if (lhs[idx].size() < nbits) {
      lhs[idx].resize(nbits);
    }
    if (rhs[idx].size() < nbits) {
      rhs[idx].resize(nbits);
    }
  });

  std::vector<std::array<bool, 2>> c(lhs.size(), {false, false});
  std::vector<conversion::BitStream> result(lhs.size());

  pforeach(0, lhs.size(), [&](uint64_t idx) { result[idx].resize(nbits + 1); });

  for (uint64_t i = 0; i < nbits; i++) {
    ArrayRef inner_lhs(makeType<spdzwisefield::BShrTy>(PT_U64, 64), size);
    ArrayRef inner_rhs(makeType<spdzwisefield::BShrTy>(PT_U64, 64), size);

    auto _inner_lhs = ArrayView<std::array<uint64_t, 2>>(inner_lhs);
    auto _inner_rhs = ArrayView<std::array<uint64_t, 2>>(inner_rhs);

    pforeach(0, size, [&](uint64_t idx) {
      auto& lhs_val = _inner_lhs[idx];
      auto& rhs_val = _inner_rhs[idx];

      lhs_val[0] = lhs_val[1] = 0;
      rhs_val[0] = rhs_val[1] = 0;

      for (uint64_t j = 0; j < 64; j++) {
        if (idx * 64 + j >= lhs.size()) {
          break;
        }

        lhs_val[0] |=
            static_cast<uint64_t>(lhs[idx * 64 + j][i][0] ^ c[idx * 64 + j][0])
            << j;
        lhs_val[1] |=
            static_cast<uint64_t>(lhs[idx * 64 + j][i][1] ^ c[idx * 64 + j][1])
            << j;

        rhs_val[0] |=
            static_cast<uint64_t>(rhs[idx * 64 + j][i][0] ^ c[idx * 64 + j][0])
            << j;
        rhs_val[1] |=
            static_cast<uint64_t>(rhs[idx * 64 + j][i][1] ^ c[idx * 64 + j][1])
            << j;

        result[idx * 64 + j][i][0] = lhs[idx * 64 + j][i][0] ^
                                     rhs[idx * 64 + j][i][0] ^
                                     c[idx * 64 + j][0];

        result[idx * 64 + j][i][1] = lhs[idx * 64 + j][i][1] ^
                                     rhs[idx * 64 + j][i][1] ^
                                     c[idx * 64 + j][1];
      }
    });

    // ArrayRef _open1 = ctx->call("b2p", inner_lhs);
    // ArrayRef _open2 = ctx->call("b2p", inner_rhs);

    // auto open1 = ArrayView<uint64_t>(_open1);
    // auto open2 = ArrayView<uint64_t>(_open2);

    ArrayRef res;
    if (with_check) {
      res = ctx->call("and_bb", inner_lhs, inner_rhs);
    } else {
      res = semi_honest_and_bb(ctx, inner_lhs, inner_rhs);
    }

    // ArrayRef _open3 = ctx->call("b2p", res);
    // auto open3 = ArrayView<uint64_t>(_open3);

    // if (comm->getRank() == 0) {
    //   std::cout << open1[0] << " " << open2[0] << " " << open3[0] <<
    //   std::endl;
    // }

    auto _res = ArrayView<std::array<uint64_t, 2>>(res);

    pforeach(0, size, [&](uint64_t idx) {
      auto res_val = _res[idx];

      for (uint64_t j = 0; j < 64; j++) {
        if (idx * 64 + j >= lhs.size()) {
          break;
        }
        c[idx * 64 + j][0] ^= (res_val[0] >> j) & 1;
        c[idx * 64 + j][1] ^= (res_val[1] >> j) & 1;
      }
    });
  }

  pforeach(0, lhs.size(), [&](uint64_t idx) {
    result[idx][nbits][0] = c[idx][0];
    result[idx][nbits][1] = c[idx][1];
  });

  return result;
}

std::vector<bool> open_bits(Object* ctx, const conversion::BitStream& bts) {
  auto* comm = ctx->getState<Communicator>();
  (void)comm;

  size_t size = (bts.size() - 1) / 64 + 1;
  std::vector<bool> result(bts.size());

  ArrayRef inner_bts(makeType<spdzwisefield::BShrTy>(PT_U64, 64), size);
  auto _inner_bts = ArrayView<std::array<uint64_t, 2>>(inner_bts);

  pforeach(0, size, [&](uint64_t idx) {
    auto& bts_val = _inner_bts[idx];

    bts_val[0] = bts_val[1] = 0;

    for (uint64_t j = 0; j < 64; j++) {
      if (idx * 64 + j >= bts.size()) {
        break;
      }

      bts_val[0] |= static_cast<uint64_t>(bts[idx * 64 + j][0]) << j;
      bts_val[1] |= static_cast<uint64_t>(bts[idx * 64 + j][1]) << j;
    }
  });

  ArrayRef res = ctx->call("b2p", inner_bts);
  auto _res = ArrayView<uint64_t>(res);

  pforeach(0, size, [&](uint64_t idx) {
    auto res_val = _res[idx];

    for (uint64_t j = 0; j < 64; j++) {
      if (idx * 64 + j >= bts.size()) {
        break;
      }
      result[idx * 64 + j] = (res_val >> j) & 1;
    }
  });

  return result;
}

bool check_edabits(const std::vector<conversion::PubEdabit>& edabits) {
  pforeach(0, edabits.size(), [&](uint64_t idx) {
    const auto& edabit = edabits[idx];
    auto arith = edabit.dataA;
    auto bits = edabit.dataB;

    uint64_t tmp = 0;
    for (int i = 0; i < 61; i++) {
      tmp = tmp + (static_cast<uint64_t>(bits[i]) << i);
    }

    SPU_ENFORCE(arith == tmp, "edabit check failed");
  });

  return true;
}

}  // namespace spu::mpc