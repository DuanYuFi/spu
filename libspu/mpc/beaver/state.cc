
#include "libspu/mpc/beaver/state.h"

#include "libspu/core/array_ref.h"
#include "libspu/mpc/beaver/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"

namespace spu::mpc {

/*
 TODO: Not finished since the cut and choose in arithmetic is not implemented
 yet.
*/

/*
void BeaverState::gen_arith_triples(
    Object* ctx, std::vector<beaver::ArithmeticTriple>* triples,
    size_t num_triples, size_t batch_size = BeaverState::batch_size_,
    size_t bucket_size = BeaverState::bucket_size_) {
  if (triples->size() >= num_triples || num_triples == 0) {
    return;
  }

  auto* comm = ctx->getState<Communicator>();
  auto* prg = ctx->getState<PrgState>();

  size_t rank = comm->getRank();

  size_t triples_needed = num_triples - triples->size();
  size_t num_per_batch = batch_size * bucket_size + bucket_size;
  size_t num_batches = (triples_needed + num_per_batch - 1) / num_per_batch;

  std::vector<beaver::ArithmeticTriple> new_triples(num_per_batch *
                                                    num_batches);

  if (rank == 0) {
    std::vector<beaver::ring2k_t> r1(num_per_batch * num_batches),
        r2(num_per_batch * num_batches);

    std::vector<beaver::ring2k_t> share_a(num_per_batch * num_batches),
        share_b(num_per_batch * num_batches),
        share_c(num_per_batch * num_batches);

    std::vector<beaver::ring2k_t> buffer_next(num_per_batch * num_batches * 3);

    prg->fillPriv(absl::MakeSpan(r1));
    prg->fillPriv(absl::MakeSpan(r2));
    prg->fillPrssPair(absl::MakeSpan(share_a), absl::MakeSpan(share_a), false,
                      true);
    prg->fillPrssPair(absl::MakeSpan(share_b), absl::MakeSpan(share_b), false,
                      true);
    prg->fillPrssPair(absl::MakeSpan(share_c), absl::MakeSpan(share_c), false,
                      true);

    pforeach(0, num_batches * num_per_batch, [&](uint64_t idx) {
      new_triples[idx][0][0] = share_a[idx];
      new_triples[idx][0][1] = r1[idx] - share_a[idx];
      new_triples[idx][1][0] = share_b[idx];
      new_triples[idx][1][1] = r2[idx] - share_b[idx];
      new_triples[idx][2][0] = share_c[idx];
      new_triples[idx][2][1] = r1[idx] * r2[idx] - share_c[idx];

      buffer_next[idx * 3] = new_triples[idx][0][1];
      buffer_next[idx * 3 + 1] = new_triples[idx][1][1];
      buffer_next[idx * 3 + 2] = new_triples[idx][2][1];
    });

    comm->sendAsync<beaver::ring2k_t>(1, buffer_next, "cut-and-choose send");
  }

  else if (rank == 1) {
    std::vector<beaver::ring2k_t> buffer_prev =
        comm->recv<beaver::ring2k_t>(0, "cut-and-choose recv");

    pforeach(0, num_batches * num_per_batch, [&](uint64_t idx) {
      new_triples[idx][0][0] = buffer_prev[idx * 3];
      new_triples[idx][1][0] = buffer_prev[idx * 3 + 1];
      new_triples[idx][2][0] = buffer_prev[idx * 3 + 2];

      new_triples[idx][0][1] = 0;
      new_triples[idx][1][1] = 0;
      new_triples[idx][2][1] = 0;
    });
  }

  else {
    std::vector<beaver::ring2k_t> share_a(num_per_batch * num_batches),
        share_b(num_per_batch * num_batches),
        share_c(num_per_batch * num_batches);

    prg->fillPrssPair(absl::MakeSpan(share_a), absl::MakeSpan(share_a), true,
                      false);
    prg->fillPrssPair(absl::MakeSpan(share_b), absl::MakeSpan(share_b), true,
                      false);
    prg->fillPrssPair(absl::MakeSpan(share_c), absl::MakeSpan(share_c), true,
                      false);

    pforeach(0, num_batches * num_per_batch, [&](uint64_t idx) {
      new_triples[idx][0][1] = share_a[idx];
      new_triples[idx][1][1] = share_b[idx];
      new_triples[idx][2][1] = share_c[idx];

      new_triples[idx][0][0] = 0;
      new_triples[idx][1][0] = 0;
      new_triples[idx][2][0] = 0;
    });
  }
}

*/

// TODO: deal with return value and caller function.

ArrayRef BeaverState::gen_bin_triples(
    Object* ctx, PtType out_type, size_t size,
    size_t batch_size = BeaverState::batch_size_,
    size_t bucket_size = BeaverState::bucket_size_) {
  auto* comm = ctx->getState<Communicator>();
  auto* prg = ctx->getState<PrgState>();

  size_t rank = comm->getRank();

  int compress =
      sizeof(typename beaver::BTDataType) / SizeOf(PtTypeToField(out_type));
  int need = (size - 1) / compress + 1;
  int64_t triples_needed = need - trusted_triples_bin_->size();

  if (triples_needed > 0) {
    size_t num_per_batch = batch_size * bucket_size + bucket_size;
    size_t num_batches = (triples_needed + num_per_batch - 1) / num_per_batch;

    std::vector<beaver::BinaryTriple> new_triples(num_per_batch * num_batches);

    if (rank == 0) {
      std::vector<beaver::BTDataType> r1(num_per_batch * num_batches),
          r2(num_per_batch * num_batches);

      std::vector<beaver::BTDataType> share_a(num_per_batch * num_batches),
          share_b(num_per_batch * num_batches),
          share_c(num_per_batch * num_batches);

      std::vector<beaver::BTDataType> buffer_next(num_per_batch * num_batches *
                                                  3);

      prg->fillPriv(absl::MakeSpan(r1));
      prg->fillPriv(absl::MakeSpan(r2));
      prg->fillPrssPair(absl::MakeSpan(share_a), absl::MakeSpan(share_a), false,
                        true);
      prg->fillPrssPair(absl::MakeSpan(share_b), absl::MakeSpan(share_b), false,
                        true);
      prg->fillPrssPair(absl::MakeSpan(share_c), absl::MakeSpan(share_c), false,
                        true);

      pforeach(0, num_batches * num_per_batch, [&](uint64_t idx) {
        new_triples[idx][0][0] = share_a[idx];
        new_triples[idx][0][1] = r1[idx] ^ share_a[idx];
        new_triples[idx][1][0] = share_b[idx];
        new_triples[idx][1][1] = r2[idx] ^ share_b[idx];
        new_triples[idx][2][0] = share_c[idx];
        new_triples[idx][2][1] = (r1[idx] & r2[idx]) ^ share_c[idx];

        buffer_next[idx * 3] = new_triples[idx][0][1];
        buffer_next[idx * 3 + 1] = new_triples[idx][1][1];
        buffer_next[idx * 3 + 2] = new_triples[idx][2][1];
      });

      comm->sendAsync<beaver::BTDataType>(1, buffer_next,
                                          "cut-and-choose send");
    }

    else if (rank == 1) {
      std::vector<beaver::BTDataType> buffer_prev =
          comm->recv<beaver::BTDataType>(0, "cut-and-choose recv");

      pforeach(0, num_batches * num_per_batch, [&](uint64_t idx) {
        new_triples[idx][0][0] = buffer_prev[idx * 3];
        new_triples[idx][1][0] = buffer_prev[idx * 3 + 1];
        new_triples[idx][2][0] = buffer_prev[idx * 3 + 2];

        new_triples[idx][0][1] = 0;
        new_triples[idx][1][1] = 0;
        new_triples[idx][2][1] = 0;
      });
    }

    else {
      std::vector<beaver::BTDataType> share_a(num_per_batch * num_batches),
          share_b(num_per_batch * num_batches),
          share_c(num_per_batch * num_batches);

      prg->fillPrssPair(absl::MakeSpan(share_a), absl::MakeSpan(share_a), true,
                        false);
      prg->fillPrssPair(absl::MakeSpan(share_b), absl::MakeSpan(share_b), true,
                        false);
      prg->fillPrssPair(absl::MakeSpan(share_c), absl::MakeSpan(share_c), true,
                        false);

      pforeach(0, num_batches * num_per_batch, [&](uint64_t idx) {
        new_triples[idx][0][1] = share_a[idx];
        new_triples[idx][1][1] = share_b[idx];
        new_triples[idx][2][1] = share_c[idx];

        new_triples[idx][0][0] = 0;
        new_triples[idx][1][0] = 0;
        new_triples[idx][2][0] = 0;
      });
    }

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

    ArrayRef out(makeType<beaver::BinTripleTy>(out_type), size);
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

    return out;
  });
}

/*
 TODO: Triple verif. using another without opening in arithmetic is not
 mentioned in FLNW17, so this part is to be continued.
*/

/*
std::vector<beaver::ArithmeticTriple> BeaverState::cut_and_choose(
    Object* ctx, typename std::vector<beaver::ArithmeticTriple>::iterator data,
    size_t batch_size, size_t bucket_size, size_t C) {
  auto* comm = ctx->getState<Communicator>();
  auto* prg = ctx->getState<PrgState>();

  auto seedRef = ctx->call("A2P", ctx->call("RandA", 1));
  auto seed = ArrayView<beaver::ring2k_t>(seedRef)[0];

  size_t num_per_batch = batch_size * bucket_size + C;
  std::mt19937 rng(seed);
  std::shuffle(data, data + num_per_batch, rng);

  ArrayRef buf(makeType<beaver::Rss3PC>(), C * 3);

  auto _buf = ArrayView<beaver::Rss3PC>(buf);
  pforeach(0, C, [&](uint64_t idx) {
    _buf[idx * 3] = data[idx][0];
    _buf[idx * 3 + 1] = data[idx][1];
    _buf[idx * 3 + 2] = data[idx][2];
  });

  auto opened = ctx->call("A2P", buf);

  auto _opened = ArrayView<beaver::ring2k_t>(opened);
  pforeach(0, C, [&](uint64_t idx) {
    beaver::ring2k_t a = _opened[idx * 3];
    beaver::ring2k_t b = _opened[idx * 3 + 1];
    beaver::ring2k_t c = _opened[idx * 3 + 2];

    assert(a * b == c);
  });
}
*/

// Refer to:
// 2.10 Triple Verification Using Another (Without Opening)
// FLNW17 (High-Throughput Secure Three-Party Computation for Malicious
// Adversaries and an Honest Majority)

std::vector<beaver::BinaryTriple> BeaverState::cut_and_choose(
    Object* ctx, typename std::vector<beaver::BinaryTriple>::iterator data,
    size_t batch_size, size_t bucket_size, size_t C) {
  auto* comm = ctx->getState<Communicator>();
  auto* prg = ctx->getState<PrgState>();

  // generate random seed for shuffle

  auto seedRef = ctx->call("A2P", ctx->call("RandA", 1));
  auto seed = ArrayView<beaver::BTDataType>(seedRef)[0];

  size_t num_per_batch = batch_size * bucket_size + C;
  std::mt19937 rng(seed);
  std::shuffle(data, data + num_per_batch, rng);

  // open first C triples

  ArrayRef buf(makeType<beaver::BinRss3PC>(), C * 3);
  auto _buf = ArrayView<beaver::BinRss3PC>(buf);
  pforeach(0, C, [&](uint64_t idx) {
    _buf[idx * 3] = data[idx][0];
    _buf[idx * 3 + 1] = data[idx][1];
    _buf[idx * 3 + 2] = data[idx][2];
  });

  auto opened = ctx->call("B2P", buf);

  auto _opened = ArrayView<beaver::BTDataType>(opened);
  pforeach(0, C, [&](uint64_t idx) {
    beaver::BTDataType a = _opened[idx * 3];
    beaver::BTDataType b = _opened[idx * 3 + 1];
    beaver::BTDataType c = _opened[idx * 3 + 2];

    assert(a & b == c);
  });

  data += C;

  // PROTOCOL 2.24 (Triple Verif. Using Another Without Opening)

  ArrayRef buf2(makeType<beaver::BinRss3PC>(),
                batch_size * (bucket_size - 1) * 2);

  auto _buf2 = ArrayView<beaver::BinRss3PC>(buf2);

  pforeach(0, batch_size, [&](uint64_t idx) {
    for (size_t i = 0; i < bucket_size - 1; i++) {
      _buf2[idx * (bucket_size - 1) * 2 + i * 2][0] =
          data[idx * bucket_size][0][0] ^ data[idx * bucket_size + i + 1][0][0];
      _buf2[idx * (bucket_size - 1) * 2 + i * 2][1] =
          data[idx * bucket_size][0][1] ^ data[idx * bucket_size + i + 1][0][1];

      _buf2[idx * (bucket_size - 1) * 2 + i * 2 + 1][0] =
          data[idx * bucket_size][1][0] ^ data[idx * bucket_size + i + 1][1][0];
      _buf2[idx * (bucket_size - 1) * 2 + i * 2 + 1][1] =
          data[idx * bucket_size][1][1] ^ data[idx * bucket_size + i + 1][1][1];
    }
  });

  auto opened2 = ctx->call("B2P", buf2);
  auto _opened2 = ArrayView<beaver::BTDataType>(opened2);

  std::vector<beaver::BTDataType> z(batch_size * (bucket_size - 1));

  std::vector<beaver::BTDataType> buf3(batch_size * (bucket_size - 1));

  pforeach(0, batch_size, [&](uint64_t idx) {
    beaver::BinaryTriple trusted = data[idx * bucket_size];
    for (size_t i = 0; i < bucket_size - 1; i++) {
      auto rho = _opened2[idx * (bucket_size - 1) * 2 + i * 2];
      auto sigma = _opened2[idx * (bucket_size - 1) * 2 + i * 2 + 1];

      beaver::BinaryTriple untrusted = data[idx * bucket_size + i + 1];

      auto x1 = trusted[2][0] ^ untrusted[2][0] ^ sigma & untrusted[0][0] ^
                rho & untrusted[1][0] ^ rho & sigma;
      auto x2 = trusted[2][1] ^ untrusted[2][1] ^ sigma & untrusted[0][1] ^
                rho & untrusted[1][1] ^ rho & sigma;

      z[idx * (bucket_size - 1) + i] = x1;
      buf3[idx * (bucket_size - 1) + i] = x2;
    }
  });

  auto z2 = comm->rotate<beaver::BTDataType>(buf3, "TripleVerify");

  pforeach(0, batch_size * (bucket_size - 1),
           [&](uint64_t idx) { assert(z[idx] == z2[idx]); });

  std::vector<beaver::BinaryTriple> res(batch_size);

  pforeach(0, batch_size,
           [&](uint64_t idx) { res[idx] = data[idx * bucket_size]; });

  return res;
}

}  // namespace spu::mpc