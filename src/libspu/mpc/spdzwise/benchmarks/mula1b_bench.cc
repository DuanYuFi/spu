// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// https://www.apache.org/licenses/LICENSE-2.0

#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/aby3/arithmetic.h"
#include "libspu/mpc/aby3/io.h"
#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/spdzwise/io.h"
#include "libspu/mpc/spdzwise/protocol.h"
#include "libspu/mpc/spdzwise/ring_arithmetic.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdzwise {
// Observation only. Operations always use the normally registered kernel.
class SpdzWiseBenchmarkAccess {
 public:
  static auto times(const SpdzWiseState* s) { return s->mula1b_ms_; }
};
namespace {
using Clock = std::chrono::steady_clock;
class Barrier {
  std::mutex mutex_;
  std::condition_variable cv_;
  size_t arrived_ = 0, generation_ = 0;

 public:
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto generation = generation_;
    if (++arrived_ == 3) {
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
    } else
      cv_.wait(lock, [&] { return generation_ != generation; });
  }
};
uint64_t SplitMix(uint64_t& state) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
struct Result {
  double total = 0, operation = 0, final_check = 0;
  std::array<double, 4> phases{};
  size_t bytes = 0, actions = 0, checks = 0, input_checks = 0,
         boolean_checks = 0;
  NdArrayRef output;
};
void Run(const std::string& protocol, int64_t n, int reps) {
  const bool wise = protocol == "spdzwise";
  NdArrayRef plain(makeType<RingTy>(FM64), {n});
  NdArrayRef bits(makeType<RingTy>(FM64), {n});
  uint64_t seed = 0x4d554c413142ULL;
  const std::array<uint64_t, 6> edges{
      0, 1, UINT64_MAX, uint64_t(1) << 63, (uint64_t(1) << 63) - 1, 65536};
  for (int64_t i = 0; i < n; ++i) {
    plain.data<uint64_t>()[i] =
        size_t(i / 2) < edges.size() ? edges[i / 2] : SplitMix(seed);
    bits.data<uint64_t>()[i] =
        size_t(i / 2) < edges.size() ? (i & 1) : (SplitMix(seed) & 1);
  }
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  auto aby_io = aby3::makeAby3Io(FM64, 3);
  RuntimeConfig cfg(wise ? SPDZWISE : ABY3, FM64);
  cfg.spdzwise_config.domain = SpdzWiseConfig::RING;
  cfg.experimental_enable_inter_op_par = false;
  cfg.experimental_enable_intra_op_par = false;
  std::cout
      << std::fixed << std::setprecision(6)
      << "protocol,count,iteration,rank,total_ms,operation_ms,final_check_ms,"
         "dabit_ms,precheck_ms,bit2a_ms,multiply_ms,sent_bytes,sent_actions,"
         "input_checks,mac_checks,boolean_checks,outputs_verified\n";
  for (int iteration = 0; iteration <= reps; ++iteration) {
    const auto inputs = wise ? io.toShares(plain, VIS_SECRET)
                             : aby_io->toShares(plain, VIS_SECRET, -1);
    auto bin = io.toBooleanShares(bits, 1);
    if (!wise)
      for (auto& b : bin) b = b.as(makeType<aby3::BShrTy>(PT_U64, 1));
    Barrier barrier;
    const auto results = utils::simulate(
        3, [&](const std::shared_ptr<yacl::link::Context>& link) {
          auto ctx = wise ? std::make_unique<SPUContext>(cfg, link)
                          : makeAby3Protocol(cfg, link);
          if (wise) registerArithmeticPrototype(ctx.get(), link);
          auto* s = wise ? ctx->prot()->getState<SpdzWiseState>() : nullptr;
          ProtocolRunGuard guard(ctx.get());
          Value x(inputs[link->Rank()], DT_U64), b(bin[link->Rank()], DT_I1);
          Result result;
          if (wise) {
            SPU_ENFORCE(
                dynamic_cast<ring::MulA1B*>(ctx->prot()->getKernel("mul_a1b")));
            s->begin(ExecutableProto("mula1b-bench", {"x", "b"}, {"y"}, "test"),
                     {x, b});
            x = import_s(ctx.get(), x);
            b = import_s(ctx.get(), b);
            result.input_checks = s->arithmeticChecks();
            SPU_ENFORCE(result.input_checks == 1 &&
                        s->pendingBooleanBits() == 0);
            // Empty check outside timing confirms that input MAC work was
            // flushed.
            s->verify();
            SPU_ENFORCE(s->arithmeticChecks() == result.input_checks &&
                        s->booleanChecks() == 0);
          } else {
            SPU_ENFORCE(
                dynamic_cast<aby3::MulA1B*>(ctx->prot()->getKernel("mul_a1b")));
          }
          const auto stats = link->GetStats();
          barrier.wait();
          const auto bytes = stats->sent_bytes.load();
          const auto actions = stats->sent_actions.load();
          const auto start = Clock::now();
          auto y = mul_a1b(ctx.get(), x, b);
          const auto operation_stop = Clock::now();
          if (wise) s->checkMac();
          const auto stop = Clock::now();
          result.total =
              std::chrono::duration<double, std::milli>(stop - start).count();
          result.operation =
              std::chrono::duration<double, std::milli>(operation_stop - start)
                  .count();
          result.final_check =
              std::chrono::duration<double, std::milli>(stop - operation_stop)
                  .count();
          result.bytes = stats->sent_bytes.load() - bytes;
          result.actions = stats->sent_actions.load() - actions;
          if (wise) {
            result.phases = SpdzWiseBenchmarkAccess::times(s);
            result.checks = s->arithmeticChecks() - result.input_checks;
            result.boolean_checks = s->booleanChecks();
            SPU_ENFORCE(result.checks == 2 && result.boolean_checks == 0 &&
                        s->pendingBooleanBits() == 0);
            SPU_ENFORCE(y.dtype() == x.dtype() &&
                        y.storage_type() == x.storage_type() &&
                        y.shape() == x.shape());
            s->prepareOutputs(ctx.get(), {"y"}, {y});
            guard.commit();
            SPU_ENFORCE(s->arithmeticChecks() ==
                            result.input_checks + result.checks &&
                        s->booleanChecks() == 0);
            result.output = s->exportedValue("y").data();
          } else {
            SPU_ENFORCE(y.storage_type() == x.storage_type() &&
                        y.shape() == x.shape());
            result.output = y.data();
            guard.commit();
            result.final_check = 0;
          }
          return result;
        });
    std::vector<NdArrayRef> output;
    for (const auto& r : results) output.push_back(r.output);
    const auto actual =
        wise ? io.fromShares(output) : aby_io->fromShares(output);
    for (int64_t i = 0; i < n; ++i) {
      SPU_ENFORCE(actual.data<uint64_t>()[i] ==
                      plain.data<uint64_t>()[i] * bits.data<uint64_t>()[i],
                  "{} output mismatch at {}", protocol, i);
      for (size_t rank = 0; rank < 3; ++rank)
        SPU_ENFORCE(output[rank].data<RepShare>()[i][1] ==
                    output[(rank + 1) % 3].data<RepShare>()[i][0]);
    }
    for (size_t rank = 0; rank < 3; ++rank) {
      const auto& r = results[rank];
      std::cout << protocol << ',' << n << ',' << iteration << ',' << rank
                << ',' << r.total << ',' << r.operation << ',' << r.final_check;
      for (const auto t : r.phases) std::cout << ',' << t;
      std::cout << ',' << r.bytes << ',' << r.actions << ',' << r.input_checks
                << ',' << r.checks << ',' << r.boolean_checks << ',' << n
                << '\n';
    }
    std::cout.flush();
    std::cerr << protocol << " iteration " << iteration << ": verified " << n
              << " outputs\n";
  }
}
}  // namespace
}  // namespace spu::mpc::spdzwise
int main(int argc, char** argv) {
  try {
    const std::string protocol = argc > 1 ? argv[1] : "spdzwise";
    const auto n = argc > 2 ? std::stoll(argv[2]) : 1000000;
    const auto reps = argc > 3 ? std::stoi(argv[3]) : 7;
    SPU_ENFORCE((protocol == "spdzwise" || protocol == "aby3") && n > 0 &&
                reps > 0 && argc <= 4);
    spu::mpc::spdzwise::Run(protocol, n, reps);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
