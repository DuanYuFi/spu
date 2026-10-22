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
#include "libspu/mpc/api.h"
#include "libspu/mpc/spdzwise/io.h"
#include "libspu/mpc/spdzwise/protocol.h"
#include "libspu/mpc/spdzwise/ring_truncation.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdzwise {
namespace {
using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point start, Clock::time_point end = Clock::now()) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}
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
    } else {
      cv_.wait(lock, [&] { return generation_ != generation; });
    }
  }
};

uint64_t SplitMix(uint64_t& state) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
NdArrayRef Inputs(int64_t count) {
  NdArrayRef out(makeType<RingTy>(FM64), {count});
  constexpr uint64_t bound = uint64_t(1) << 62;
  const std::array<uint64_t, 9> edges{0,
                                      1,
                                      7,
                                      65535,
                                      65536,
                                      65537,
                                      bound - 1,
                                      bound - 65536,
                                      uint64_t(1) << 40};
  uint64_t seed = 0x5350445a5452554eULL;
  for (int64_t i = 0; i < count; i += 2) {
    const uint64_t magnitude = size_t(i / 2) < edges.size()
                                   ? edges[i / 2]
                                   : 1 + SplitMix(seed) % (bound - 1);
    out.data<uint64_t>()[i] = magnitude;
    out.data<uint64_t>()[i + 1] = uint64_t(0) - magnitude;
  }
  return out;
}
struct Result {
  double total_ms = 0, material_ms = 0, online_ms = 0, verify_ms = 0;
  size_t bytes = 0, actions = 0, verify_bytes = 0, input_checks = 0,
         final_checks = 0;
  NdArrayRef output;
};

void Run(bool wise, int64_t count, int repetitions) {
  const auto plain = Inputs(count);
  SpdzWiseIo wise_io(FM64, 3, SpdzWiseConfig::RING);
  auto aby_io = aby3::makeAby3Io(FM64, 3);
  RuntimeConfig config(wise ? SPDZWISE : ABY3, FM64);
  config.spdzwise_config.domain = SpdzWiseConfig::RING;
  config.fxp_fraction_bits = 16;
  std::cout
      << std::fixed << std::setprecision(6)
      << "protocol,count,bits,iteration,rank,total_ms,material_ms,online_ms,"
         "verify_ms,sent_bytes,sent_actions,verify_bytes,input_checks,"
         "final_checks,outputs_verified,error_minus1,error_zero,error_plus1\n";
  for (int iteration = 0; iteration <= repetitions; ++iteration) {
    const auto inputs = wise ? wise_io.toShares(plain, VIS_SECRET)
                             : aby_io->toShares(plain, VIS_SECRET, -1);
    Barrier barrier;
    const auto results = utils::simulate(3, [&](const auto& link) {
      std::unique_ptr<SPUContext> ctx =
          wise ? std::make_unique<SPUContext>(config, link)
               : makeAby3Protocol(config, link);
      if (wise) registerArithmeticPrototype(ctx.get(), link);
      auto* state = wise ? ctx->prot()->getState<SpdzWiseState>() : nullptr;
      ProtocolRunGuard guard(ctx.get());
      Value x(inputs[link->Rank()], DT_I64);
      Result result;
      if (wise) {
        SPU_ENFORCE(
            dynamic_cast<ring::TruncASelector*>(ctx->prot()->getKernel("trunc_a")));
        state->begin(ExecutableProto("truncation-bench", {"x"}, {"y"}, "test"),
                     {x});
        x = import_s(ctx.get(), x);
        result.input_checks = state->arithmeticChecks();
        SPU_ENFORCE(result.input_checks == 1);
      } else {
        SPU_ENFORCE(
            dynamic_cast<aby3::TruncAPr2*>(ctx->prot()->getKernel("trunc_a")),
            "benchmark requires the normally registered ABY3 TruncAPr2");
      }
      const auto stats = link->GetStats();
      barrier.wait();
      const auto bytes = stats->sent_bytes.load();
      const auto actions = stats->sent_actions.load();
      const auto start = Clock::now();
      auto y = trunc_a(ctx.get(), x, 16, SignType::Unknown);
      const auto stop = Clock::now();  // Output shares exist; stop immediately.
      result.total_ms = Ms(start, stop);
      result.bytes = stats->sent_bytes.load() - bytes;
      result.actions = stats->sent_actions.load() - actions;
      result.material_ms = wise ? state->truncMaterialMs() : 0;
      result.online_ms = result.total_ms - result.material_ms;
      SPU_ENFORCE(y.shape() == x.shape() &&
                  y.storage_type() == x.storage_type());
      if (wise) {
        SPU_ENFORCE(state->arithmeticChecks() == result.input_checks,
                    "unexpected final MAC check inside truncation timing");
        barrier.wait();
        const auto verify_bytes = stats->sent_bytes.load();
        const auto verify_start = Clock::now();
        state->checkMac();
        result.verify_ms = Ms(verify_start);
        result.verify_bytes = stats->sent_bytes.load() - verify_bytes;
        result.final_checks = state->arithmeticChecks() - result.input_checks;
        SPU_ENFORCE(result.final_checks == 1);
        y.setDtype(DT_I64);
        state->prepareOutputs(ctx.get(), {"y"}, {y});
        guard.commit();
        result.output = state->exportedValue("y").data();
      } else {
        result.output = y.data();
        guard.commit();
      }
      return result;
    });
    std::vector<NdArrayRef> output;
    for (const auto& r : results) output.push_back(r.output);
    const auto reconstructed =
        wise ? wise_io.fromShares(output) : aby_io->fromShares(output);
    std::array<size_t, 3> errors{};
    for (int64_t i = 0; i < count; ++i) {
      const auto x = static_cast<int64_t>(plain.data<uint64_t>()[i]);
      const auto y = static_cast<int64_t>(reconstructed.data<uint64_t>()[i]);
      const auto floor = x / 65536 - (x % 65536 < 0);
      const auto error = y - floor;
      SPU_ENFORCE(error >= (wise ? 0 : -1) && error <= 1,
                  "wrong output: iteration {}, element {}, x {}, y {}",
                  iteration, i, x, y);
      if (wise && x % 65536 == 0) SPU_ENFORCE(error == 0);
      ++errors[error + 1];
    }
    std::cerr << "iteration " << iteration << ": all " << count
              << " outputs verified; errors (-1,0,+1)=" << errors[0] << ','
              << errors[1] << ',' << errors[2] << '\n';
    for (size_t rank = 0; rank < 3; ++rank) {
      const auto& r = results[rank];
      std::cout << (wise ? "spdzwise-ring-signed" : "aby3-truncapr2") << ','
                << count << ",16," << iteration << ',' << rank << ','
                << r.total_ms << ',' << r.material_ms << ',' << r.online_ms
                << ',' << r.verify_ms << ',' << r.bytes << ',' << r.actions
                << ',' << r.verify_bytes << ',' << r.input_checks << ','
                << r.final_checks << ',' << count << ',' << errors[0] << ','
                << errors[1] << ',' << errors[2] << '\n';
    }
    std::cout.flush();
  }
}
}  // namespace
}  // namespace spu::mpc::spdzwise

int main(int argc, char** argv) {
  try {
    const std::string protocol = argc > 1 ? argv[1] : "spdzwise";
    SPU_ENFORCE(protocol == "spdzwise" || protocol == "aby3");
    const int64_t count = argc > 2 ? std::stoll(argv[2]) : 100000;
    const int repetitions = argc > 3 ? std::stoi(argv[3]) : 7;
    SPU_ENFORCE(count >= 20 && count % 2 == 0 && repetitions > 0);
    spu::mpc::spdzwise::Run(protocol == "spdzwise", count, repetitions);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
