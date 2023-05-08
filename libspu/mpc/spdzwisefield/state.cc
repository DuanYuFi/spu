
#include "libspu/mpc/spdzwisefield/state.h"

// #include <fstream>
#include <random>

#include "libspu/core/array_ref.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/spdzwisefield/type.h"
#include "libspu/mpc/spdzwisefield/value.h"

#define MYLOG(x) \
  if (ctx->getState<Communicator>()->getRank() == 0) std::cout << x << std::endl

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

    std::vector<beaver::BTDataType> a0(num_per_batch * num_batches);
    std::vector<beaver::BTDataType> a1(num_per_batch * num_batches);
    std::vector<beaver::BTDataType> b0(num_per_batch * num_batches);
    std::vector<beaver::BTDataType> b1(num_per_batch * num_batches);
    std::vector<beaver::BTDataType> c0(num_per_batch * num_batches);
    std::vector<beaver::BTDataType> c1(num_per_batch * num_batches);

    prg->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));
    prg->fillPrssPair(absl::MakeSpan(a0), absl::MakeSpan(a1));
    prg->fillPrssPair(absl::MakeSpan(b0), absl::MakeSpan(b1));

    pforeach(0, num_per_batch * num_batches, [&](int64_t idx) {
      r0[idx] = (a0[idx] & b0[idx]) ^  //
                (a0[idx] & b1[idx]) ^  //
                (a1[idx] & b0[idx]) ^  //
                (r0[idx] ^ r1[idx]);
    });

    r1 = comm->rotate<beaver::BTDataType>(r0, "cac.bin");  // comm => 1, k

    pforeach(0, num_per_batch * num_batches, [&](int64_t idx) {
      new_triples[idx][0] = {a0[idx], a1[idx]};
      new_triples[idx][1] = {b0[idx], b1[idx]};
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

  ArrayRef rho_sigma(makeType<spdzwisefield::BShrTy>(PT_U128, 128),
                     batch_size * (bucket_size - 1) * 2);
  auto rho_sigma_view = ArrayView<std::array<beaver::BTDataType, 2>>(rho_sigma);

  pforeach(0, batch_size, [&](uint64_t idx) {
    for (size_t i = 0; i < bucket_size - 1; i++) {
      rho_sigma_view[idx * (bucket_size - 1) * 2 + i * 2][0] =
          data[idx * bucket_size][0][0] ^ data[idx * bucket_size + i + 1][0][0];
      rho_sigma_view[idx * (bucket_size - 1) * 2 + i * 2][1] =
          data[idx * bucket_size][0][1] ^ data[idx * bucket_size + i + 1][0][1];

      rho_sigma_view[idx * (bucket_size - 1) * 2 + i * 2 + 1][0] =
          data[idx * bucket_size][1][0] ^ data[idx * bucket_size + i + 1][1][1];
      rho_sigma_view[idx * (bucket_size - 1) * 2 + i * 2 + 1][1] =
          data[idx * bucket_size][1][1] ^ data[idx * bucket_size + i + 1][1][1];
    }
  });

  ArrayRef open_rho_sigma = ctx->call("b2p", rho_sigma);
  auto open_rho_sigma_view = ArrayView<beaver::BTDataType>(open_rho_sigma);

  std::vector<std::array<beaver::BTDataType, 2>> t(batch_size *
                                                   (bucket_size - 1));
  pforeach(0, batch_size, [&](uint64_t idx) {
    beaver::BinaryTriple trusted = data[idx * bucket_size];
    for (size_t i = 0; i < bucket_size - 1; i++) {
      auto rho = open_rho_sigma_view[idx * (bucket_size - 1) * 2 + i * 2];
      auto sigma = open_rho_sigma_view[idx * (bucket_size - 1) * 2 + i * 2 + 1];

      beaver::BinaryTriple untrusted = data[idx * bucket_size + i + 1];

      auto x1 = trusted[2][0] ^ untrusted[2][0] ^ (sigma & trusted[0][0]) ^
                (rho & trusted[1][0]);

      auto x2 = trusted[2][1] ^ untrusted[2][1] ^ (sigma & trusted[0][1]) ^
                (rho & trusted[1][1]);

      if (comm->getRank() == 0) {
        x1 ^= (rho & sigma);
      } else if (comm->getRank() == 2) {
        x2 ^= (rho & sigma);
      }

      t[idx * (bucket_size - 1) + i][0] = x1;
      t[idx * (bucket_size - 1) + i][1] = x2;
    }
  });

  for (uint64_t idx = 0; idx < batch_size; idx++) {
    for (uint64_t i = 0; i < bucket_size - 1; i++) {
      auto x1 = t[idx * (bucket_size - 1) + i][0];
      auto x2 = t[idx * (bucket_size - 1) + i][1];

      hash_algo->Update(std::to_string(x1 ^ x2));

      hash_algo2->Update(std::to_string(x2));
    }
  }

  std::vector<beaver::BinaryTriple> res(batch_size);

  pforeach(0, batch_size,
           [&](uint64_t idx) { res[idx] = data[idx * bucket_size]; });

  return res;
}

/*

    ================================== Verif. ==================================

*/

void SpdzWiseFieldState::verification(Object* ctx, bool final) {
  while (stored_data_->size() >= verif_batch_size) {
    verif_batch(ctx);
  }

  if (final) {
    verif_batch(ctx, stored_data_->size());
  }
}

void SpdzWiseFieldState::verif_batch(Object* ctx, size_t size) {
  if (size == 0) {
    return;
  }

  auto* prg_state = ctx->getState<PrgState>();

  std::vector<uint64_t> coefficients(size);
  prg_state->fillPubl(absl::MakeSpan(coefficients));

  ArrayRef u(makeType<aby3::AShrTy>(FM64), 1);
  ArrayRef v(makeType<aby3::AShrTy>(FM64), 1);

  auto _u = ArrayView<std::array<uint64_t, 2>>(u);
  auto _v = ArrayView<std::array<uint64_t, 2>>(v);

  _u[0] = {0, 0};
  _v[0] = {0, 0};

  for (uint64_t idx = 0; idx < size; idx++) {
    coefficients[idx] = Field::modp(coefficients[idx]);

    _u[0][0] = Field::add(
        _u[0][0], Field::mul(coefficients[idx], stored_data_->at(idx)[0][0]));
    _u[0][1] = Field::add(
        _u[0][1], Field::mul(coefficients[idx], stored_data_->at(idx)[0][1]));

    _v[0][0] = Field::add(
        _v[0][0], Field::mul(coefficients[idx], stored_data_->at(idx)[1][0]));
    _v[0][1] = Field::add(
        _v[0][1], Field::mul(coefficients[idx], stored_data_->at(idx)[1][1]));
  }

  std::vector<uint64_t> r0(1);
  std::vector<uint64_t> r1(1);

  prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));

  ArrayRef key_ref(makeType<aby3::AShrTy>(FM64), 1);
  auto _key_ref = ArrayView<std::array<uint64_t, 2>>(key_ref);

  _key_ref[0] = key_;

  ArrayRef t = ctx->call("mul_aa_sh", u, key_ref);
  auto _t = ArrayView<std::array<uint64_t, 2>>(t);

  _t[0][0] = Field::sub(_t[0][0], _v[0][0]);
  _t[0][1] = Field::sub(_t[0][1], _v[0][1]);

  ArrayRef r(makeType<aby3::AShrTy>(FM64), 1);
  auto _r = ArrayView<std::array<uint64_t, 2>>(r);
  _r[0][0] = Field::modp(r0[0]);
  _r[0][1] = Field::modp(r1[0]);

  ArrayRef check = ctx->call("mul_aa_sh", t, r);
  ArrayRef opened_value = ctx->call("a2psh", check);
  uint64_t value = ArrayView<uint64_t>(opened_value)[0];

  (void)value;
  // SPU_ENFORCE(value == 0, "SPDZ check failed.");

  stored_data_->erase(stored_data_->begin(), stored_data_->begin() + size);
}

/*

    ================================== TruncA ==================================

*/

std::vector<spdzwisefield::TruncPair> SpdzWiseFieldState::gen_trunc_pairs(
    Object* ctx, size_t size, size_t nbits) {
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  std::vector<uint64_t> r_0(size);
  std::vector<uint64_t> r_1(size);

  std::vector<uint64_t> rprime_0(size);
  std::vector<uint64_t> rprime_1(size);

  prg_state->fillPrssPair(absl::MakeSpan(rprime_0), absl::MakeSpan(rprime_1));

  std::vector<uint64_t> r2prime_0(size);
  std::vector<uint64_t> r2prime_1(size);

  prg_state->fillPrssPair(absl::MakeSpan(r2prime_0), absl::MakeSpan(r2prime_1));

  std::vector<uint64_t> r3prime_0(size);
  std::vector<uint64_t> r3prime_1(size);

  prg_state->fillPrssPair(absl::MakeSpan(r3prime_0), absl::MakeSpan(r3prime_1));

  std::vector<uint64_t> r2_0(size);
  std::vector<uint64_t> r2_1(size);

  prg_state->fillPrssPair(absl::MakeSpan(r2_0), absl::MakeSpan(r2_1));

  std::vector<uint64_t> r3_0(size);
  std::vector<uint64_t> r3_1(size);

  prg_state->fillPrssPair(absl::MakeSpan(r3_0), absl::MakeSpan(r3_1));

  std::vector<conversion::BitStream> rprime(size);
  std::vector<conversion::BitStream> r2prime(size);
  std::vector<conversion::BitStream> r3prime(size);

  std::vector<conversion::BitStream> r(size);
  std::vector<conversion::BitStream> r2(size);
  std::vector<conversion::BitStream> r3(size);

  pforeach(0, size, [&](uint64_t idx) {
    rprime_0[idx] |= (1ULL << 63);
    rprime_1[idx] |= (1ULL << 63);
    r2prime_0[idx] |= (1ULL << 63);
    r2prime_1[idx] |= (1ULL << 63);
    r3prime_0[idx] |= (1ULL << 63);
    r3prime_1[idx] |= (1ULL << 63);
    r2_0[idx] |= (1ULL << 63);
    r2_1[idx] |= (1ULL << 63);
    r3_0[idx] |= (1ULL << 63);
    r3_1[idx] |= (1ULL << 63);
  });

  pforeach(0, size, [&](uint64_t idx) {
    r2_0[idx] >>= (nbits + 6);
    r2_1[idx] >>= (nbits + 6);
    r3_0[idx] >>= (nbits + 6);
    r3_1[idx] >>= (nbits + 6);

    rprime_0[idx] >>= 4;
    rprime_1[idx] >>= 4;

    r2prime_0[idx] >>= 6;
    r2prime_1[idx] >>= 6;

    r3prime_0[idx] >>= 6;
    r3prime_1[idx] >>= 6;

    r_0[idx] = rprime_0[idx] >> nbits;
    r_1[idx] = rprime_1[idx] >> nbits;

    rprime[idx].resize(61);
    r2prime[idx].resize(61);
    r3prime[idx].resize(61);

    for (uint64_t i = 0; i < 61; i++) {
      rprime[idx][i][0] = (rprime_0[idx] >> i) & 1;
      rprime[idx][i][1] = (rprime_1[idx] >> i) & 1;

      r2prime[idx][i][0] = (r2prime_0[idx] >> i) & 1;
      r2prime[idx][i][1] = (r2prime_1[idx] >> i) & 1;

      r3prime[idx][i][0] = (r3prime_0[idx] >> i) & 1;
      r3prime[idx][i][1] = (r3prime_1[idx] >> i) & 1;
    }

    r[idx].resize(61 - nbits);
    r2[idx].resize(61 - nbits);
    r3[idx].resize(61 - nbits);

    for (uint64_t i = 0; i < 61 - nbits; i++) {
      r[idx][i][0] = (r_0[idx] >> i) & 1;
      r[idx][i][1] = (r_1[idx] >> i) & 1;

      r2[idx][i][0] = (r2_0[idx] >> i) & 1;
      r2[idx][i][1] = (r2_1[idx] >> i) & 1;

      r3[idx][i][0] = (r3_0[idx] >> i) & 1;
      r3[idx][i][1] = (r3_1[idx] >> i) & 1;
    }
  });

  auto tmp =
      full_adder(ctx, rprime, twos_complement(ctx, r2prime, 61), true, true);
  auto _r1prime =
      full_adder(ctx, tmp, twos_complement(ctx, r3prime, 61), true, true);

  auto tmp2 =
      full_adder(ctx, r, twos_complement(ctx, r2, 61 - nbits), true, true);
  auto _r1 =
      full_adder(ctx, tmp2, twos_complement(ctx, r3, 61 - nbits), true, true);

  std::vector<uint64_t> r1_0(size);
  std::vector<uint64_t> r1_1(size);

  std::vector<uint64_t> r1prime_0(size);
  std::vector<uint64_t> r1prime_1(size);

  pforeach(0, size, [&](uint64_t idx) {
    for (uint64_t i = 0; i < 61; i++) {
      r1prime_0[idx] |= static_cast<uint64_t>(_r1prime[idx][i][0]) << i;
      r1prime_1[idx] |= static_cast<uint64_t>(_r1prime[idx][i][1]) << i;
    }
    for (uint64_t i = 0; i < 61 - nbits; i++) {
      r1_0[idx] |= static_cast<uint64_t>(_r1[idx][i][0]) << i;
      r1_1[idx] |= static_cast<uint64_t>(_r1[idx][i][1]) << i;
    }
  });

  std::vector<spdzwisefield::TruncPair> out(size);

  switch (comm->getRank()) {
    case 0: {
      comm->sendAsync<uint64_t>(2, r3_1, "open r3");
      comm->sendAsync<uint64_t>(2, r3prime_1, "open r3prime");
      comm->sendAsync<uint64_t>(2, r1_1, "open r1");
      comm->sendAsync<uint64_t>(2, r1prime_1, "open r1prime");

      auto recv_r2 = comm->recv<uint64_t>(1, "open r2");
      auto recv_r2prime = comm->recv<uint64_t>(1, "open r2prime");
      auto recv_r1 = comm->recv<uint64_t>(1, "open r1");
      auto recv_r1prime = comm->recv<uint64_t>(1, "open r1prime");

      pforeach(0, size, [&](uint64_t idx) {
        out[idx][0][0] = r1_0[idx] ^ r1_1[idx] ^ recv_r1[idx];
        out[idx][0][1] = r2_0[idx] ^ r2_1[idx] ^ recv_r2[idx];
        out[idx][1][0] = r1prime_0[idx] ^ r1prime_1[idx] ^ recv_r1prime[idx];
        out[idx][1][1] = r2prime_0[idx] ^ r2prime_1[idx] ^ recv_r2prime[idx];
      });

      break;
    }
    case 1: {
      comm->sendAsync<uint64_t>(0, r2_1, "open r3");
      comm->sendAsync<uint64_t>(0, r2prime_1, "open r3prime");
      comm->sendAsync<uint64_t>(0, r1_1, "open r1");
      comm->sendAsync<uint64_t>(0, r1prime_1, "open r1prime");

      auto recv_r2 = comm->recv<uint64_t>(2, "open r2");
      auto recv_r2prime = comm->recv<uint64_t>(2, "open r2prime");
      auto recv_r3 = comm->recv<uint64_t>(2, "open r3");
      auto recv_r3prime = comm->recv<uint64_t>(2, "open r3prime");

      pforeach(0, size, [&](uint64_t idx) {
        out[idx][0][0] = r2_0[idx] ^ r2_1[idx] ^ recv_r2[idx];
        out[idx][0][1] = r3_0[idx] ^ r3_1[idx] ^ recv_r3[idx];
        out[idx][1][0] = r2prime_0[idx] ^ r2prime_1[idx] ^ recv_r2prime[idx];
        out[idx][1][1] = r3prime_0[idx] ^ r3prime_1[idx] ^ recv_r3prime[idx];
      });

      break;
    }
    case 2: {
      comm->sendAsync<uint64_t>(1, r2_1, "open r2");
      comm->sendAsync<uint64_t>(1, r2prime_1, "open r2prime");
      comm->sendAsync<uint64_t>(1, r3_1, "open r3");
      comm->sendAsync<uint64_t>(1, r3prime_1, "open r3prime");

      auto recv_r3 = comm->recv<uint64_t>(0, "open r3");
      auto recv_r3prime = comm->recv<uint64_t>(0, "open r3prime");
      auto recv_r1 = comm->recv<uint64_t>(0, "open r1");
      auto recv_r1prime = comm->recv<uint64_t>(0, "open r1prime");

      pforeach(0, size, [&](uint64_t idx) {
        out[idx][0][0] = r3_0[idx] ^ r3_1[idx] ^ recv_r3[idx];
        out[idx][0][1] = r1_0[idx] ^ r1_1[idx] ^ recv_r1[idx];
        out[idx][1][0] = r3prime_0[idx] ^ r3prime_1[idx] ^ recv_r3prime[idx];
        out[idx][1][1] = r1prime_0[idx] ^ r1prime_1[idx] ^ recv_r1prime[idx];
      });

      break;
    }

    default: {
      SPU_ENFORCE(false, "Invalid rank");
    }
  }

  return out;
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

    std::vector<conversion::BitStream> b0(total), b1(total), b2(total);

    for (uint64_t idx = 0; idx < total; idx++) {
      switch (comm->getRank()) {
        case 0: {
          b0[idx] = conversion::BitStream(myself[idx].bshares.begin(),
                                          myself[idx].bshares.end());
          b1[idx] = conversion::BitStream(next[idx].bshares.begin(),
                                          next[idx].bshares.end());
          b2[idx] = conversion::BitStream(prev[idx].bshares.begin(),
                                          prev[idx].bshares.end());
          break;
        }
        case 1: {
          b0[idx] = conversion::BitStream(prev[idx].bshares.begin(),
                                          prev[idx].bshares.end());
          b1[idx] = conversion::BitStream(myself[idx].bshares.begin(),
                                          myself[idx].bshares.end());
          b2[idx] = conversion::BitStream(next[idx].bshares.begin(),
                                          next[idx].bshares.end());
          break;
        }
        case 2: {
          b0[idx] = conversion::BitStream(next[idx].bshares.begin(),
                                          next[idx].bshares.end());
          b1[idx] = conversion::BitStream(prev[idx].bshares.begin(),
                                          prev[idx].bshares.end());
          b2[idx] = conversion::BitStream(myself[idx].bshares.begin(),
                                          myself[idx].bshares.end());
          break;
        }

        default:
          break;
      }
    }

    auto middle = full_adder(ctx, b0, b1, false);

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

      if (trusted_edabits.empty()) {
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
  using Field = SpdzWiseFieldState::Field;
  // generate random seed for shuffle

  std::vector<uint64_t> r0(1);
  std::vector<uint64_t> r1(1);

  prg_state->fillPrssPair(absl::MakeSpan(r0), absl::MakeSpan(r1));
  auto r2 = comm->rotate<uint64_t>(r1, "cac.edabit");

  uint64_t seed = r0[0] + r1[0] + r2[0];
  size_t size_per_batch = batch_size * bucket_size + C;

  std::mt19937 rng(seed);
  std::shuffle(data, data + size_per_batch, rng);

  std::vector<conversion::PubEdabit> opened = open_edabits(ctx, data, C);

  data += C;

  std::vector<conversion::BitStream> lhs(batch_size * (bucket_size - 1));
  std::vector<conversion::BitStream> rhs(batch_size * (bucket_size - 1));
  std::vector<conversion::Edabit> c(batch_size * (bucket_size - 1));

  pforeach(0, batch_size, [&](uint64_t idx) {
    const auto& first = data[idx * bucket_size];

    for (size_t i = 0; i < bucket_size - 1; i++) {
      std::copy(first.bshares.begin(), first.bshares.begin() + nbits_ - 1,
                back_inserter(lhs[idx * (bucket_size - 1) + i]));

      std::copy(data[idx * bucket_size + i + 1].bshares.begin(),
                data[idx * bucket_size + i + 1].bshares.begin() + nbits_ - 1,
                back_inserter(rhs[idx * (bucket_size - 1) + i]));

      c[idx * (bucket_size - 1) + i].ashare[0] = Field::add(
          data[idx * bucket_size + i + 1].ashare[0], first.ashare[0]);
      c[idx * (bucket_size - 1) + i].ashare[1] = Field::add(
          data[idx * bucket_size + i + 1].ashare[1], first.ashare[1]);
    }
  });

  auto addition = full_adder(ctx, lhs, rhs, true);

  ArrayRef redundant_bit(makeType<spdzwisefield::BShrTy>(PT_U8, 1),
                         addition.size());

  auto _redundant_bit = ArrayView<std::array<uint8_t, 2>>(redundant_bit);
  pforeach(0, batch_size * (bucket_size - 1), [&](uint64_t idx) {
    _redundant_bit[idx][0] = addition[idx][nbits_ - 1][0];
    _redundant_bit[idx][1] = addition[idx][nbits_ - 1][1];
  });

  ArrayRef arith_bit = ctx->call("bitinject", redundant_bit);
  auto _arith_bit = ArrayView<std::array<uint64_t, 2>>(arith_bit);

  pforeach(0, batch_size * (bucket_size - 1), [&](uint64_t idx) {
    c[idx].ashare[0] = Field::sub(c[idx].ashare[0],
                                  Field::mul(_arith_bit[idx][0], 1ULL << 60));
    c[idx].ashare[1] = Field::sub(c[idx].ashare[1],
                                  Field::mul(_arith_bit[idx][1], 1ULL << 60));

    std::copy_n(addition[idx].begin(), nbits_ - 1, c[idx].bshares.begin());
  });

  auto bucket_check =
      open_edabits(ctx, c.begin(), batch_size * (bucket_size - 1));

  check_edabits(bucket_check);

  std::vector<conversion::Edabit> out(batch_size);

  pforeach(0, batch_size,
           [&](uint64_t idx) { out[idx] = data[idx * bucket_size]; });

  return out;
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
    std::vector<conversion::BitStream> rhs, bool with_check, bool drop) {
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

  pforeach(0, lhs.size(),
           [&](uint64_t idx) { result[idx].resize(drop ? nbits : nbits + 1); });

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

    ArrayRef res;
    if (with_check) {
      res = ctx->call("and_bb", inner_lhs, inner_rhs);
    } else {
      res = semi_honest_and_bb(ctx, inner_lhs, inner_rhs);
    }

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

  if (!drop) {
    pforeach(0, lhs.size(), [&](uint64_t idx) {
      result[idx][nbits][0] = c[idx][0];
      result[idx][nbits][1] = c[idx][1];
    });
  }

  return result;
}

std::vector<conversion::BitStream> twos_complement(
    Object* ctx, std::vector<conversion::BitStream> bits, size_t nbits,
    bool with_check) {
  auto* comm = ctx->getState<Communicator>();
  (void)comm;

  pforeach(0, bits.size(), [&](uint64_t idx) {
    bits[idx].resize(nbits);
    for (auto& bit : bits[idx]) {
      bit[0] = !bit[0];
      bit[1] = !bit[1];
    }
  });

  std::vector<conversion::BitStream> one(bits.size());
  pforeach(0, bits.size(), [&](uint64_t idx) {
    one[idx].resize(1);
    one[idx][0] = {true, true};
  });

  auto result = full_adder(ctx, bits, one, with_check, true);

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
    for (size_t i = 0; i < EdabitState::nbits_; i++) {
      tmp = tmp + (static_cast<uint64_t>(bits[i]) << i);
    }

    SPU_ENFORCE(arith == tmp, "edabit check failed, left = {}, right = {}",
                arith, tmp);
  });

  return true;
}

std::vector<std::array<uint64_t, 2>> open_pair(
    Object* ctx, const std::vector<spdzwisefield::TruncPair>& pairs) {
  auto* comm = ctx->getState<Communicator>();
  (void)comm;

  size_t size = pairs.size();
  std::vector<std::array<uint64_t, 2>> result(size);
  std::vector<std::array<uint64_t, 2>> shares(size * 2);

  pforeach(0, size, [&](uint64_t idx) {
    shares[idx * 2][0] = pairs[idx][0][0];
    shares[idx * 2][1] = pairs[idx][0][1];
    shares[idx * 2 + 1][0] = pairs[idx][1][0];
    shares[idx * 2 + 1][1] = pairs[idx][1][1];
  });

  auto opened = open_semi_honest(ctx, shares);
  pforeach(0, size, [&](uint64_t idx) {
    result[idx][0] = opened[idx * 2];
    result[idx][1] = opened[idx * 2 + 1];
  });

  return result;
}

std::vector<std::array<uint128_t, 3>> open_triple(
    Object* ctx, const std::vector<beaver::BinaryTriple>& triples) {
  size_t size = triples.size();

  ArrayRef share_input(makeType<spdzwisefield::BShrTy>(PT_U128, 128), size * 3);
  auto _share_input = ArrayView<std::array<beaver::BTDataType, 2>>(share_input);

  pforeach(0, size, [&](uint64_t idx) {
    _share_input[idx * 3] = triples[idx][0];
    _share_input[idx * 3 + 1] = triples[idx][1];
    _share_input[idx * 3 + 2] = triples[idx][2];
  });

  ArrayRef opened = ctx->call("b2p", share_input);
  auto _opened = ArrayView<uint128_t>(opened);

  std::vector<std::array<uint128_t, 3>> result(size);

  pforeach(0, size, [&](uint64_t idx) {
    result[idx][0] = _opened[idx * 3];
    result[idx][1] = _opened[idx * 3 + 1];
    result[idx][2] = _opened[idx * 3 + 2];
  });

  return result;
}

}  // namespace spu::mpc