// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/spdzwise/ring.h"

#include <random>

#include "gtest/gtest.h"
#include "yacl/crypto/hash/blake3.h"

#include "libspu/mpc/api.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/io.h"
#include "libspu/mpc/spdzwise/protocol.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdzwise::ring {
namespace {
RuntimeConfig Config() {
  RuntimeConfig config(SPDZWISE, FM64);
  config.spdzwise_config.domain = SpdzWiseConfig::RING;
  config.fxp_fraction_bits = 16;
  return config;
}
NdArrayRef Plain(const Shape& shape) {
  NdArrayRef result(makeType<RingTy>(FM64), shape);
  const std::array<uint64_t, 8> edge = {0,
                                        1,
                                        uint64_t(-1),
                                        uint64_t(-2),
                                        uint64_t(1) << 63,
                                        (uint64_t(1) << 63) - 1,
                                        0xffffffff,
                                        0x100000001};
  for (int64_t i = 0; i < result.numel(); ++i) {
    result.data<uint64_t>()[i] = edge[i % edge.size()];
  }
  return result;
}
std::array<RepShare, 3> Split(Word value) {
  const Word a = (Word(0xfeedbeef12345678ULL) << 64) | 0xffffffffffffffffULL;
  const Word b = (Word(0x123456789abcdef0ULL) << 64) | 0xfedcba9876543210ULL;
  const Word c = value - a - b;
  return {RepShare{a, b}, RepShare{b, c}, RepShare{c, a}};
}

void ExpectCheckFailure(const std::function<void()>& action) {
  try {
    action();
    FAIL() << "expected a protocol rejection";
  } catch (const std::exception& error) {
    EXPECT_NE(std::string(error.what()).find("SPDZWISE ring check failed"),
              std::string::npos)
        << error.what();
  }
}

TEST(SpdzWiseRing, WireEncodingIsFixedLittleEndian) {
  const std::vector<Word> words = {
      0, Word(-1), Word(1) << 64,
      (Word(0x1122334455667788ULL) << 64) | 0x99aabbccddeeff00ULL};
  const auto bytes = Encode(words);
  ASSERT_EQ(bytes.size(), 64);
  EXPECT_EQ(static_cast<uint8_t>(bytes[48]), 0);
  EXPECT_EQ(static_cast<uint8_t>(bytes[49]), 0xff);
  EXPECT_EQ(static_cast<uint8_t>(bytes[63]), 0x11);
  EXPECT_EQ(Decode(bytes, words.size()), words);
  EXPECT_THROW(Decode(bytes.substr(1), words.size()), std::exception);
  EXPECT_THROW(Decode(bytes + "x", words.size()), std::exception);
  EXPECT_TRUE(Decode("", 0).empty());
}

TEST(SpdzWiseRing, Protocol9EveryNonzeroBitAndCarry) {
  utils::simulate(3, [&](const auto& link) {
    for (size_t bit = 0; bit < 128; ++bit) {
      const auto shares = Split(Word(1) << bit);
      ExpectCheckFailure([&] {
        CheckZero(link, shares[link->Rank()],
                  "test/session/execution/batch/" + std::to_string(bit));
      });
    }
    const auto zero = Split(0);
    EXPECT_NO_THROW(CheckZero(link, zero[link->Rank()], "test/zero/carry"));
    std::mt19937_64 rng(20261021);  // Reproducible test shares, not PRSS.
    for (size_t batch = 0; batch < 16; ++batch) {
      const Word a = (Word(rng()) << 64) | rng();
      const Word b = (Word(rng()) << 64) | rng();
      const Word c = Word(0) - a - b;
      const std::array<RepShare, 3> shares = {RepShare{a, b}, RepShare{b, c},
                                              RepShare{c, a}};
      EXPECT_NO_THROW(CheckZero(link, shares[link->Rank()],
                                "test/random/" + std::to_string(batch)));
    }
    // All-zero and random-looking consistent sharings both accept.
    EXPECT_NO_THROW(CheckZero(link, {0, 0}, "test/zero/plain"));
    for (size_t corrupt = 0; corrupt < 3; ++corrupt) {
      auto copy = zero[link->Rank()];
      if (link->Rank() == corrupt) copy[1] += Word(1) << 64;
      ExpectCheckFailure([&] {
        CheckZero(link, copy, "test/copy/" + std::to_string(corrupt));
      });
    }
  });
}

TEST(SpdzWiseRing, IoTypesFullRangeAndStrides) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING), field(FM64, 3);
  for (const Shape& shape : {Shape{0}, Shape{4, 8}}) {
    const auto raw = Plain(shape);
    const auto input = raw.ndim() == 2 ? raw.transpose({1, 0}) : raw;
    const auto shares = io.toShares(input, VIS_SECRET);
    const auto restored = io.fromShares(shares);
    const NdArrayView<uint64_t> a(input), b(restored);
    for (int64_t i = 0; i < input.numel(); ++i) {
      EXPECT_EQ(a[i], b[i]);
      Word lifted = 0;
      for (const auto& share : shares)
        lifted += NdArrayView<RepShare>(share)[i][0];
      EXPECT_EQ(lifted, Word(a[i]));
    }
    EXPECT_EQ(shares[0].elsize(), 32);
    EXPECT_EQ(Type::fromString(shares[0].eltype().toString()),
              shares[0].eltype());
    EXPECT_ANY_THROW(field.fromShares(shares));
  }
  EXPECT_FALSE(makeType<AShrTy>("session").isSerializable());
  Value private_value(NdArrayRef(makeType<AShrTy>("session"), {1}), DT_I64);
  EXPECT_ANY_THROW(private_value.toMetaProto());
  EXPECT_ANY_THROW(private_value.toProto(1024));
  EXPECT_ANY_THROW(Type::fromString("spdzwise.ring.AShr<forged>"));
  AShrTy internal;
  EXPECT_ANY_THROW(internal.fromString("session"));
  EXPECT_NE(makeType<InputTy>(0), makeType<spdzwise::InputTy>(0));
  EXPECT_NE(makeType<OutputTy>(0, std::string(64, 'a')),
            makeType<spdzwise::OutputTy>(0, std::string(64, 'a')));
  auto shares = io.toShares(Plain({1}), VIS_SECRET);
  shares[1].data<RepShare>()[0][0]++;
  EXPECT_ANY_THROW(io.fromShares(shares));
  EXPECT_ANY_THROW(io.toShares(Plain({1}), VIS_SECRET, 0));
}

TEST(SpdzWiseRing, ArithmeticChainingInternal128AndChecked64Export) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  const auto raw = Plain({4, 8}).transpose({1, 0});
  const auto shares = io.toShares(raw, VIS_SECRET);
  const auto results = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Value input(shares[link->Rank()], DT_I64);
    ProtocolRunGuard guard(&ctx);
    state->begin(ExecutableProto("ring-ops", {"x"}, {"y"}, "test"), {input});
    auto x = import_s(&ctx, input);
    x = Value(x.data().transpose({1, 0}).clone().transpose({1, 0}), DT_I64);
    auto pub = Value(raw.as(makeType<Pub2kTy>(FM64)), DT_I64);
    auto y = dynDispatch(&ctx, "mul_aa", x, x);
    y = dynDispatch(&ctx, "mul_aa", y, y);  // full 128-bit overflow
    y = dynDispatch(&ctx, "add_ap", y, pub);
    y = dynDispatch(&ctx, "negate_a", y);
    y = dynDispatch(&ctx, "mul_ap", y, pub);
    auto p = dynDispatch(&ctx, "p2a", pub);
    y = dynDispatch(&ctx, "add_aa", y, p);
    EXPECT_ANY_THROW(state->exportValue(
        y.data(), makeType<OutputTy>(link->Rank(), std::string(64, 'a'))));
    EXPECT_ANY_THROW(state->validateInternal(
        y.data().as(makeType<AShrTy>("different-session"))));
    EXPECT_ANY_THROW(state->validateInternal(
        NdArrayRef(makeType<spdzwise::AShrTy>(state->session()), {1})));
    // Shape operations preserve each complete 64-byte authenticated element.
    y = Value(y.data().transpose({1, 0}).clone().transpose({1, 0}), DT_I64);
    state->prepareOutputs(&ctx, {"y"}, {y});
    guard.commit();
    return std::make_pair(y.data(), state->exportedValue("y").data());
  });
  std::vector<NdArrayRef> external;
  for (const auto& result : results) {
    external.push_back(result.second);
    EXPECT_EQ(result.first.elsize(), 64);
    EXPECT_EQ(result.second.elsize(), 16);
  }
  const auto restored = io.fromShares(external);
  const NdArrayView<uint64_t> input(raw);
  for (int64_t i = 0; i < raw.numel(); ++i) {
    Word sum = 0;
    for (const auto& result : results)
      sum += NdArrayView<AuthShare>(result.first)[i][0];
    const Word x = input[i];
    const Word expected = (Word(0) - (x * x * x * x + x)) * x + x;
    EXPECT_EQ(sum, expected);
    EXPECT_EQ(restored.data<uint64_t>()[i], static_cast<uint64_t>(expected));
  }
}

TEST(SpdzWiseRing, MatrixSecretAndPublicWithStrides) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  for (int64_t k : {0, 1, 7, 16}) {
    const auto a = Plain({k, 2}).transpose({1, 0});
    const auto b = Plain({3, k}).transpose({1, 0});
    const auto shares_a = io.toShares(a, VIS_SECRET);
    const auto shares_b = io.toShares(b, VIS_SECRET);
    const auto results = utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* state = ctx.prot()->getState<SpdzWiseState>();
      Value x(shares_a[link->Rank()], DT_I64),
          y(shares_b[link->Rank()], DT_I64);
      state->begin(
          ExecutableProto("ring-matrix", {"a", "b"}, {"s", "p"}, "test"),
          {x, y});
      x = import_s(&ctx, x);
      y = import_s(&ctx, y);
      auto before = link->GetStats()->sent_bytes.load();
      auto secret = dynDispatch(&ctx, "mmul_aa", x, y);
      const auto bytes = link->GetStats()->sent_bytes.load() - before;
      // Six output cells; the wire includes a fixed malformed-message status.
      EXPECT_EQ(bytes, 6 * 32 + 4);
      auto public_result = dynDispatch(
          &ctx, "mmul_ap", x, Value(b.as(makeType<Pub2kTy>(FM64)), DT_I64));
      state->prepareOutputs(&ctx, {"s", "p"}, {secret, public_result});
      state->commit();
      return std::make_pair(state->exportedValue("s").data(),
                            state->exportedValue("p").data());
    });
    std::vector<NdArrayRef> secret, public_result;
    for (auto& r : results) {
      secret.push_back(r.first);
      public_result.push_back(r.second);
    }
    const auto s = io.fromShares(secret), p = io.fromShares(public_result);
    const NdArrayView<uint64_t> av(a), bv(b);
    for (int64_t i = 0; i < 2; ++i)
      for (int64_t j = 0; j < 3; ++j) {
        uint64_t expected = 0;
        for (int64_t t = 0; t < k; ++t)
          expected += av[i * k + t] * bv[t * 3 + j];
        EXPECT_EQ(s.data<uint64_t>()[i * 3 + j], expected);
        EXPECT_EQ(p.data<uint64_t>()[i * 3 + j], expected);
      }
  }
}

TEST(SpdzWiseRing, EmptyFinalizationReuseAndReauthentication) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  const auto empty = io.toShares(Plain({0}), VIS_SECRET);
  utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    Value input(empty[link->Rank()], DT_I64);
    state->begin(ExecutableProto("empty", {"x"}, {}, "test"), {input});
    auto x = import_s(&ctx, input);
    x = dynDispatch(&ctx, "mul_aa", x, x);
    state->prepareOutputs(&ctx, {}, {});
    state->commit();
    EXPECT_EQ(state->arithmeticChecks(), 0);
    state->begin(ExecutableProto("reuse", {"x"}, {"y"}, "test"), {x});
    auto internal = import_s(&ctx, x);
    EXPECT_EQ(state->arithmeticChecks(), 0);
    auto p = state->publicValue(Plain({8}).as(makeType<Pub2kTy>(FM64)));
    auto product = state->multiply(p, p);
    state->prepareOutputs(&ctx, {"y"}, {Value(product, DT_I64)});
    state->commit();
    const auto output = state->exportedValue("y");
    auto checks = state->arithmeticChecks();
    state->begin(ExecutableProto("reimport", {"y"}, {}, "test"), {output});
    const auto authenticated = import_s(&ctx, output);
    EXPECT_EQ(state->arithmeticChecks(), checks + 1);
    EXPECT_TRUE(authenticated.storage_type().isa<AShrTy>());
    state->multiply(authenticated.data(), authenticated.data());
    state->prepareOutputs(&ctx, {}, {});
    state->commit();
    EXPECT_EQ(state->arithmeticChecks(), checks + 2);
    (void)internal;
  });
}

TEST(SpdzWiseRing, InputCopyErrorsAbortAllRanks) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  for (size_t corrupt = 0; corrupt < 3; ++corrupt) {
    for (size_t component = 0; component < 2; ++component) {
      auto shares = io.toShares(Plain({3}), VIS_SECRET);
      shares[corrupt].data<RepShare>()[0][component] += Word(1) << 64;
      utils::simulate(3, [&](const auto& link) {
        SPUContext ctx(Config(), link);
        registerArithmeticPrototype(&ctx, link);
        auto* state = ctx.prot()->getState<SpdzWiseState>();
        const Value input(shares[link->Rank()], DT_I64);
        state->begin(ExecutableProto("bad-input", {"x"}, {}, "test"), {input});
        ExpectCheckFailure([&] { import_s(&ctx, input); });
        EXPECT_TRUE(state->failed());
        EXPECT_EQ(state->arithmeticChecks(), 0);
        EXPECT_ANY_THROW(state->commit());
      });
    }
  }
}

TEST(SpdzWiseRing, ConsistentValueAndMacErrorsFailWithoutOutputs) {
  for (size_t owner = 0; owner < 3; ++owner) {
    for (size_t component : {0U, 2U}) {
      utils::simulate(3, [&](const auto& link) {
        SPUContext ctx(Config(), link);
        registerArithmeticPrototype(&ctx, link);
        auto* state = ctx.prot()->getState<SpdzWiseState>();
        state->begin(ExecutableProto("consistent-error", {}, {}, "test"), {});
        auto plain = Plain({1}).as(makeType<Pub2kTy>(FM64));
        plain.data<uint64_t>()[0] = 3;
        auto x = state->publicValue(plain);
        auto y = state->publicValue(plain);
        // Test harness corrupts BOTH replicas to isolate the MAC relation
        // check from the preceding replicated-copy check.
        if (link->Rank() == owner) x.data<AuthShare>()[0][component]++;
        if (link->NextRank() == owner) x.data<AuthShare>()[0][component + 1]++;
        state->multiply(x, y);
        ExpectCheckFailure([&] { state->prepareOutputs(&ctx, {}, {}); });
        EXPECT_TRUE(state->failed());
        EXPECT_ANY_THROW(state->commit());
      });
    }
  }
}

TEST(SpdzWiseRing, RejectsCrossSessionAndDomainInputs) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING), field(FM64, 3);
  const auto ring_inputs = io.toShares(Plain({2}), VIS_SECRET);
  const auto field_inputs = field.toShares(Plain({2}), VIS_SECRET);
  utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    EXPECT_ANY_THROW(state->validateExternalInput(
        Value(field_inputs[link->Rank()], DT_I64)));
    state->begin(ExecutableProto("first", {}, {}, "test"), {});
    auto value = Value(
        state->publicValue(Plain({2}).as(makeType<Pub2kTy>(FM64))), DT_I64);
    state->prepareOutputs(&ctx, {}, {});
    state->commit();
    SPUContext other(Config(), link);
    registerArithmeticPrototype(&other, link);
    auto* next = other.prot()->getState<SpdzWiseState>();
    EXPECT_ANY_THROW(
        next->begin(ExecutableProto("other", {"x"}, {}, "test"), {value}));
    EXPECT_TRUE(next->failed());
    RuntimeConfig field_config(SPDZWISE, FM64);
    SpdzWiseState field_state(field_config, link);
    EXPECT_ANY_THROW(field_state.validateExternalInput(
        Value(ring_inputs[link->Rank()], DT_I64)));
  });
}

// Mutate actual transport messages, without test hooks in the protocol.
class FaultChannel : public yacl::link::transport::IChannel {
  std::shared_ptr<yacl::link::transport::IChannel> inner_;
  std::function<yacl::Buffer(const std::string&, yacl::Buffer)> change_;

 public:
  FaultChannel(
      std::shared_ptr<yacl::link::transport::IChannel> inner,
      std::function<yacl::Buffer(const std::string&, yacl::Buffer)> change)
      : inner_(std::move(inner)), change_(std::move(change)) {}
  void SendAsync(const std::string& k, yacl::Buffer v) override {
    inner_->SendAsync(k, change_(k, std::move(v)));
  }
  void SendAsyncThrottled(const std::string& k, yacl::Buffer v) override {
    inner_->SendAsyncThrottled(k, change_(k, std::move(v)));
  }
  void Send(const std::string& k, yacl::ByteContainerView v) override {
    auto b = change_(k, yacl::Buffer(v));
    inner_->Send(k, yacl::ByteContainerView(b));
  }
  yacl::Buffer Recv(const std::string& k) override { return inner_->Recv(k); }
  void SetRecvTimeout(uint64_t t) override { inner_->SetRecvTimeout(t); }
  uint64_t GetRecvTimeout() const override { return inner_->GetRecvTimeout(); }
  void WaitLinkTaskFinish() override { inner_->WaitLinkTaskFinish(); }
  void Abort() override { inner_->Abort(); }
  void SetThrottleWindowSize(size_t s) override {
    inner_->SetThrottleWindowSize(s);
  }
  void TestSend(uint32_t t) override { inner_->TestSend(t); }
  void TestRecv() override { inner_->TestRecv(); }
  void SetChunkParallelSendSize(size_t s) override {
    inner_->SetChunkParallelSendSize(s);
  }
};
std::shared_ptr<yacl::link::Context> Decorate(
    const std::shared_ptr<yacl::link::Context>& base,
    std::function<yacl::Buffer(size_t, yacl::Buffer)> change) {
  yacl::link::ContextDesc desc;
  desc.id = base->Id();
  std::vector<std::shared_ptr<yacl::link::transport::IChannel>> channels(3);
  for (size_t i = 0; i < 3; ++i) {
    desc.parties.push_back({std::to_string(i), "memory"});
    if (i != base->Rank())
      channels[i] = std::make_shared<FaultChannel>(
          base->GetChannel(i),
          [change, i](const std::string&, yacl::Buffer bytes) {
            return change(i, std::move(bytes));
          });
  }
  auto link = std::make_shared<yacl::link::Context>(desc, base->Rank(),
                                                    channels, nullptr);
  link->SetRecvTimeout(5000);
  return link;
}

TEST(SpdzWiseRing, StreamedCopyHashesMatchCanonicalTranscript) {
  utils::simulate(3, [&](const auto& base) {
    std::string expected;
    size_t digests = 0;
    auto link = Decorate(base, [&](size_t destination, yacl::Buffer bytes) {
      if (bytes.size() == 32) {
        EXPECT_EQ(destination, base->PrevRank());
        EXPECT_EQ(std::string_view(bytes), expected);
        ++digests;
      }
      return bytes;
    });
    for (size_t count : {0U, 1U, 4095U, 4096U, 4097U, 8193U}) {
      const std::string context = "stream/copies/" + std::to_string(count);
      std::vector<RepShare> shares(count);
      std::string words;
      for (size_t i = 0; i < count; ++i) {
        shares[i] = {(Word(base->Rank() + 1) << 100) + Word(i) * Word(-1),
                     (Word(base->NextRank() + 1) << 100) + Word(i) * Word(-1)};
        // Independent byte-wise oracle for the original complete transcript.
        for (size_t byte = 0; byte < 16; ++byte)
          words.push_back(static_cast<char>(shares[i][0] >> (8 * byte)));
      }
      std::string transcript;
      for (const auto& field :
           {std::string("SPDZWISE-RING-v1"), std::string("copies"), context,
            std::to_string(base->Rank()), words}) {
        transcript += std::to_string(field.size()) + ":" + field;
      }
      yacl::crypto::Blake3Hash hash;
      hash.Update(transcript);
      const auto digest = hash.CumulativeHash();
      expected.assign(reinterpret_cast<const char*>(digest.data()),
                      digest.size());
      const auto before = digests;
      CheckCopies(link, shares, context);
      EXPECT_EQ(digests, before + 1);
    }
  });
}

TEST(SpdzWiseRing, ParallelStridedArithmeticAndAggregation) {
  SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
  // Cross the parallel grain and hash chunk boundaries with noncompact inputs.
  const auto plain = Plain({257, 257}).transpose({1, 0});
  const auto shares = io.toShares(plain, VIS_SECRET);
  const auto outputs = utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* state = ctx.prot()->getState<SpdzWiseState>();
    ProtocolRunGuard guard(&ctx);
    Value input(shares[link->Rank()], DT_I64);
    state->begin(ExecutableProto("parallel-strided", {"x"}, {"z"}, "test"),
                 {input});
    auto x = import_s(&ctx, input);
    x = Value(x.data().transpose({1, 0}).clone().transpose({1, 0}), DT_I64);
    auto z = dynDispatch(&ctx, "mul_aa", x, x);
    z = dynDispatch(&ctx, "add_aa", z, x);
    z = dynDispatch(&ctx, "negate_a", z);
    z = dynDispatch(&ctx, "mul_ap", z,
                    Value(plain.as(makeType<Pub2kTy>(FM64)), DT_I64));
    state->prepareOutputs(&ctx, {"z"}, {z});
    guard.commit();
    return state->exportedValue("z").data();
  });
  const auto reconstructed = io.fromShares(outputs);
  const NdArrayView<uint64_t> p(plain), actual(reconstructed);
  for (int64_t i = 0; i < plain.numel(); ++i) {
    EXPECT_EQ(actual[i], (uint64_t(0) - (p[i] * p[i] + p[i])) * p[i]);
  }
}

TEST(SpdzWiseRing, WireTamperingPoisonsEverySession) {
  for (size_t party = 0; party < 3; ++party) {
    for (const std::string fault :
         {"value", "mac", "digest", "truncate", "product-truncate"}) {
      utils::simulate(3, [&](const auto& base) {
        int phase = 0;
        bool injected = false;
        // Channel keys are transport counters, not Context debug tags.
        // During checking, the only 32-byte message to NextRank is Protocol 9.
        auto link = Decorate(base, [&](size_t destination, yacl::Buffer bytes) {
          if (base->Rank() != party) return bytes;
          if (phase == 1 && fault == "product-truncate" &&
              destination == base->PrevRank() && bytes.size() == 32) {
            injected = true;
            return yacl::Buffer(bytes.data(), 31);
          }
          if (phase == 1 && (fault == "value" || fault == "mac") &&
              destination == base->PrevRank() && bytes.size() == 32) {
            injected = true;
            bytes.data<char>()[fault == "value" ? 0 : 16] ^= 1;
          }
          if (phase == 2 && destination == base->NextRank() &&
              bytes.size() == 32) {
            if (fault == "digest") {
              injected = true;
              bytes.data<char>()[0] ^= 1;
            }
            if (fault == "truncate") {
              injected = true;
              return yacl::Buffer(bytes.data(), 31);
            }
          }
          return bytes;
        });
        SPUContext ctx(Config(), link);
        registerArithmeticPrototype(&ctx, link);
        auto* state = ctx.prot()->getState<SpdzWiseState>();
        state->begin(ExecutableProto("fault", {}, {"y"}, "test"), {});
        auto p = state->publicValue(Plain({1}).as(makeType<Pub2kTy>(FM64)));
        phase = 1;
        if (fault == "product-truncate") {
          ExpectCheckFailure([&] { state->multiply(p, p); });
        } else {
          auto product = state->multiply(p, p);
          phase = 2;
          if (fault == "value") {
            // Repair the exposed buffer. Its queued snapshot must remain bad.
            product.data<AuthShare>()[0] = p.data<AuthShare>()[0];
          }
          ExpectCheckFailure([&] {
            state->prepareOutputs(&ctx, {"y"}, {Value(product, DT_I64)});
          });
        }
        if (base->Rank() == party) {
          EXPECT_TRUE(injected);
        }
        EXPECT_TRUE(state->failed());
        EXPECT_ANY_THROW(state->commit());
        EXPECT_ANY_THROW(state->exportedValue("y"));
        EXPECT_ANY_THROW(
            state->publicValue(Plain({1}).as(makeType<Pub2kTy>(FM64))));
        EXPECT_ANY_THROW(
            state->begin(ExecutableProto("again", {}, {}, "test"), {}));
      });
    }
  }
}

TEST(SpdzWiseRing, HashContextBindsBatchAndComponent) {
  for (size_t party = 0; party < 3; ++party) {
    utils::simulate(3, [&](const auto& base) {
      std::string old;
      int phase = 0;
      auto link = Decorate(base, [&](size_t destination, yacl::Buffer bytes) {
        if (base->Rank() == party && destination == base->NextRank() &&
            bytes.size() == 32) {
          if (phase == 0)
            old.assign(bytes.data<char>(), bytes.size());
          else if (phase == 1)
            return yacl::Buffer(old.data(), old.size());
          else if (phase == 2) {
            std::string input;
            auto append = [&](std::string_view x) {
              input += std::to_string(x.size()) + ":";
              input.append(x);
            };
            append("SPDZWISE-RING-v1");
            append("zero");
            append("session/execution/component");
            // Deliberately hash owner i+1 instead of the required i+2.
            append(std::to_string((base->Rank() + 1) % 3));
            append(Encode({0}));
            yacl::crypto::Blake3Hash hash;
            hash.Update(input);
            const auto digest = hash.CumulativeHash();
            return yacl::Buffer(digest.data(), digest.size());
          }
        }
        return bytes;
      });
      CheckZero(link, Split(0)[base->Rank()], "session/execution/batch0");
      if (base->Rank() == party) {
        EXPECT_EQ(old.size(), 32);
      }
      phase = 1;
      ExpectCheckFailure([&] {
        CheckZero(link, Split(0)[base->Rank()], "session/execution/batch1");
      });
      // All-zero components still have distinct domain-separated digests.
      phase = 2;
      ExpectCheckFailure(
          [&] { CheckZero(link, {0, 0}, "session/execution/component"); });
    });
  }
}
}  // namespace
}  // namespace spu::mpc::spdzwise::ring
