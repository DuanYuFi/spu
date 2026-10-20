// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>

#include "libspu/mpc/aby3/io.h"
#include "libspu/mpc/aby3/protocol.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/io.h"
#include "libspu/mpc/spdzwise/protocol.h"
#include "libspu/mpc/spdzwise/replicated.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdzwise {

// Development benchmark only: authenticate the two actual inputs, then defer
// their check until the single final check over all multiplication results.
class SpdzWiseBenchmarkAccess {
 public:
  static std::vector<RepShare> prepare(SpdzWiseState* state,
                                       const NdArrayRef& input) {
    state->validateExternalInput(Value(input, DT_I64));
    const NdArrayView<RepShare> view(input);
    std::vector<RepShare> values(input.numel());
    for (int64_t i = 0; i < input.numel(); ++i) {
      values[i] = view[i];
    }
    state->checkCopies(values);
    return values;
  }
  static NdArrayRef mac(SpdzWiseState* state,
                        const std::vector<RepShare>& values) {
    return state->authenticateInput(values,
                                    {static_cast<int64_t>(values.size())});
  }
  static size_t pending(const SpdzWiseState* state) {
    return state->pending_.size();
  }
};

namespace {

// F61 adapters share the SPDZwise replicated multiplication primitive.
// Stock ABY3 uses a power-of-two ring; its native addition is not used here.
Value Aby3Multiply(SPUContext* ctx, const Value& lhs, const Value& rhs) {
  const NdArrayView<RepShare> x(lhs.data()), y(rhs.data());
  NdArrayRef result(lhs.storage_type(), lhs.shape());
  auto* out = result.data<RepShare>();
  MultiplyF61(
      ctx->getState<PrgState>(), ctx->lctx(), lhs.numel(),
      [&](int64_t i) { return x[i]; }, [&](int64_t i) { return y[i]; },
      [&](int64_t i, const RepShare& z) { out[i] = z; }, "aby3-f61-mul");
  return Value(result, DT_I64);
}

Value Aby3Add(const Value& lhs, const Value& rhs) {
  const NdArrayView<RepShare> x(lhs.data()), y(rhs.data());
  NdArrayRef result(lhs.storage_type(), lhs.shape());
  auto* out = result.data<RepShare>();
  for (int64_t i = 0; i < lhs.numel(); ++i) {
    out[i] = {Add(x[i][0], y[i][0]), Add(x[i][1], y[i][1])};
  }
  return Value(result, DT_I64);
}

Value Aby3AddOffsets(const Value& x, size_t rank) {
  const NdArrayView<RepShare> in(x.data());
  NdArrayRef result(x.storage_type(), x.shape());
  auto* out = result.data<RepShare>();
  for (int64_t i = 0; i < x.numel(); ++i) {
    out[i] = {Add(in[i][0], rank == 0 ? i : 0),
              Add(in[i][1], rank == 2 ? i : 0)};
  }
  return Value(result, DT_I64);
}

class Barrier {
  std::mutex mutex_;
  std::condition_variable cv_;
  size_t arrived_ = 0;
  size_t generation_ = 0;

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

using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start, Clock::time_point stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

struct Sample {
  // input MAC, broadcast/public offsets, circuit (add + mul), final check
  std::array<double, 4> ms{};
  std::array<size_t, 4> bytes{};
  double mul_ms = 0;
  uint64_t checks = 0;
  size_t pending = 0;
};
struct Result {
  std::vector<Sample> samples;
  NdArrayRef output;
};

void Run(bool wise, const std::vector<NdArrayRef>& inputs,
         const NdArrayRef& plain, int64_t width, int64_t depth,
         int repetitions, bool native_ring = false) {
  SPU_ENFORCE(!native_ring || !wise);
  const auto gates = width * depth;
  RuntimeConfig config;
  config.protocol = wise ? SPDZWISE : ABY3;
  config.field = FM64;
  config.fxp_fraction_bits = 16;
  const std::string protocol = native_ring ? "aby3-ring64"
                              : wise     ? "spdzwise-f61"
                                         : "aby3-f61-adapter";
  NdArrayRef offsets(makeType<Pub2kTy>(FM64), {width});
  auto* offset_data = offsets.data<uint64_t>();
  for (int64_t i = 0; i < width; ++i) {
    offset_data[i] = i;
  }
  Barrier barrier;
  const auto results = utils::simulate(3, [&](const auto& link) {
    std::unique_ptr<SPUContext> ctx =
        wise ? std::make_unique<SPUContext>(config, link)
             : makeAby3Protocol(config, link);
    SpdzWiseState* state = nullptr;
    const Value original(inputs[link->Rank()], DT_I64);
    std::vector<RepShare> prepared;
    if (wise) {
      registerArithmeticPrototype(ctx.get(), link);
      state = ctx->prot()->getState<SpdzWiseState>();
      state->begin(ExecutableProto("circuit_bench", {"xy"}, {"z"}, "test"),
                   {original});
      prepared = SpdzWiseBenchmarkAccess::prepare(state, original.data());
    }
    const auto stats = link->GetStats();
    Result result;
    Value z;
    for (int iteration = 0; iteration <= repetitions; ++iteration) {
      z = Value();
      Value authenticated, y;
      Sample sample;
      const auto before_checks = wise ? state->arithmeticChecks() : 0;
      SPU_ENFORCE(!wise || SpdzWiseBenchmarkAccess::pending(state) == 0);
      auto phase = [&](size_t index, auto&& fn) {
        barrier.wait();
        const auto bytes = stats->sent_bytes.load();
        const auto start = Clock::now();
        fn();
        sample.ms[index] = Milliseconds(start, Clock::now());
        sample.bytes[index] = stats->sent_bytes.load() - bytes;
      };
      phase(0, [&] {
        if (wise) {
          authenticated =
              Value(SpdzWiseBenchmarkAccess::mac(state, prepared), DT_I64);
        }
      });
      if (!wise) {
        authenticated = original;
        sample.ms[0] = 0;
      }
      // Only local linear operations expand two authenticated scalars. No new
      // client inputs, MAC generation, or multiplication gates are hidden here.
      phase(1, [&] {
        z = dynDispatch(ctx.get(), "extract_slice", authenticated, Index{0},
                        Index{1}, Strides{1});
        y = dynDispatch(ctx.get(), "extract_slice", authenticated, Index{1},
                        Index{2}, Strides{1});
        z = dynDispatch(ctx.get(), "broadcast", z, Shape{width}, Axes{0});
        y = dynDispatch(ctx.get(), "broadcast", y, Shape{width}, Axes{0});
        z = (wise || native_ring)
                ? dynDispatch(ctx.get(), "add_ap", z, Value(offsets, DT_I64))
                : Aby3AddOffsets(z, link->Rank());
      });
      SPU_ENFORCE(!wise || SpdzWiseBenchmarkAccess::pending(state) == 2);
      // z[0,i] = x+i; z[l+1,i] = z[l,i] * (z[l,i]+y).
      // Every layer depends on the previous one, with width gates per layer.
      phase(2, [&] {
        for (int64_t layer = 0; layer < depth; ++layer) {
          const auto rhs = (wise || native_ring)
                               ? dynDispatch(ctx.get(), "add_aa", z, y)
                               : Aby3Add(z, y);
          const auto start = Clock::now();
          z = (wise || native_ring)
                  ? dynDispatch(ctx.get(), "mul_aa", z, rhs)
                  : Aby3Multiply(ctx.get(), z, rhs);
          sample.mul_ms += Milliseconds(start, Clock::now());
        }
      });
      if (wise) {
        sample.pending = SpdzWiseBenchmarkAccess::pending(state);
        SPU_ENFORCE(state->arithmeticChecks() == before_checks,
                    "verification occurred before the circuit finished");
        SPU_ENFORCE(sample.pending == static_cast<size_t>(gates + 2),
                    "wrong number of checked wires");
      }
      phase(3, [&] {
        if (wise) {
          state->check();
        }
      });
      if (!wise) {
        sample.ms[3] = 0;
      }
      sample.checks = wise ? state->arithmeticChecks() - before_checks : 0;
      SPU_ENFORCE(sample.checks == (wise ? 1 : 0), "wrong check count");
      SPU_ENFORCE(sample.bytes[0] == (wise ? 16 : 0));
      SPU_ENFORCE(sample.bytes[1] == 0);
      SPU_ENFORCE(sample.bytes[2] ==
                  static_cast<size_t>(gates) * (wise ? 16 : 8));
      SPU_ENFORCE(sample.bytes[3] == (wise ? 64 : 0));
      if (iteration != 0) {
        result.samples.push_back(sample);
      }
      if (link->Rank() == 0) {
        std::cerr << protocol << " iteration " << iteration << ": input MAC "
                  << sample.ms[0] << " ms, expand " << sample.ms[1]
                  << " ms, circuit " << sample.ms[2] << " ms (mul "
                  << sample.mul_ms << "), verify " << sample.ms[3] << " ms\n";
      }
    }
    z.setDtype(DT_I64);
    if (wise) {
      const auto checks = state->arithmeticChecks();
      state->prepareOutputs(ctx.get(), {"z"}, {z});
      state->commit();
      SPU_ENFORCE(state->arithmeticChecks() == checks,
                  "extra nonempty verification at export");
      result.output = state->exportedValue("z").data();
    } else {
      result.output = z.data();
    }
    return result;
  });
  std::vector<NdArrayRef> outputs;
  for (const auto& result : results) {
    outputs.push_back(result.output);
  }
  SpdzWiseIo io(FM64, 3);
  aby3::Aby3Io ring_io(FM64, 3);
  const auto reconstructed = native_ring ? ring_io.fromShares(outputs)
                                         : io.fromShares(outputs);
  const NdArrayView<uint64_t> actual(reconstructed), p(plain);
  for (int64_t i = 0; i < width; ++i) {
    auto expected = native_ring ? p[0] + static_cast<uint64_t>(i)
                                : Add(FromRing(p[0]), i);
    for (int64_t layer = 0; layer < depth; ++layer) {
      expected = native_ring ? expected * (expected + p[1])
                             : Mul(expected, Add(expected, FromRing(p[1])));
    }
    SPU_ENFORCE(actual[i] == (native_ring ? expected : ToRing(expected)),
                "wrong circuit output at {}", i);
  }
  std::cerr << protocol << ": checked all " << width << " outputs; " << gates
            << " multiplication gates, two secret input elements\n";
  for (size_t rank = 0; rank < results.size(); ++rank) {
    for (int iteration = 0; iteration < repetitions; ++iteration) {
      const auto& s = results[rank].samples[iteration];
      std::cout << protocol << ',' << width << ',' << depth << ',' << gates
                << ",2," << iteration << ',' << rank;
      for (const auto ms : s.ms) {
        std::cout << ',' << ms;
      }
      std::cout << ',' << s.mul_ms << ',' << s.ms[2] - s.mul_ms << ','
                << s.ms[2] + s.ms[3] << ','
                << s.ms[0] + s.ms[1] + s.ms[2] + s.ms[3];
      for (const auto bytes : s.bytes) {
        std::cout << ',' << bytes;
      }
      std::cout << ',' << s.checks << ',' << s.pending << '\n';
    }
  }
}

}  // namespace
}  // namespace spu::mpc::spdzwise

int main(int argc, char** argv) {
  using namespace spu;
  const int64_t width = argc > 1 ? std::stoll(argv[1]) : 65536;
  const int64_t depth = argc > 2 ? std::stoll(argv[2]) : 16;
  const int repetitions = argc > 3 ? std::stoi(argv[3]) : 7;
  SPU_ENFORCE(argc <= 5 && (argc < 5 || std::string(argv[4]) == "ring64"),
              "usage: circuit_bench [width depth repetitions [ring64]]");
  const bool native_ring = argc == 5;
  SPU_ENFORCE(width > 0 && depth > 0 && repetitions > 0 &&
                  width <= static_cast<int64_t>(mpc::spdzwise::kHalfPrime) &&
                  depth <= (std::numeric_limits<int64_t>::max() - 2) / width,
              "positive, nonoverflowing arguments required");
  NdArrayRef plain(makeType<RingTy>(FM64), {2});
  auto* p = plain.data<uint64_t>();
  p[0] = mpc::spdzwise::kHalfPrime - 7;
  p[1] = static_cast<uint64_t>(int64_t{-13});
  mpc::spdzwise::SpdzWiseIo io(FM64, 3);
  mpc::aby3::Aby3Io ring_io(FM64, 3);
  const auto inputs = native_ring ? ring_io.toShares(plain, VIS_SECRET, -1)
                                  : io.toShares(plain, VIS_SECRET);
  std::cout << std::fixed << std::setprecision(6)
            << "protocol,width,depth,gates,secret_inputs,iteration,rank,"
               "input_mac_ms,expand_ms,circuit_ms,verify_ms,mul_ms,local_ms,"
               "core_ms,total_ms,input_mac_bytes,expand_bytes,circuit_bytes,"
               "verify_bytes,checks,pending\n";
  mpc::spdzwise::Run(false, inputs, plain, width, depth, repetitions, native_ring);
  if (!native_ring) {
    mpc::spdzwise::Run(true, inputs, plain, width, depth, repetitions);
  }
}
