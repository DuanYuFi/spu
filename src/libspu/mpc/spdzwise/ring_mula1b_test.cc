// Copyright 2026 The SPU Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// https://www.apache.org/licenses/LICENSE-2.0

#include <cstring>
#include <random>

#include "gtest/gtest.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/api.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/spdzwise/correlated_randomness.h"
#include "libspu/mpc/spdzwise/io.h"
#include "libspu/mpc/spdzwise/protocol.h"
#include "libspu/mpc/spdzwise/ring_arithmetic.h"
#include "libspu/mpc/spdzwise/state.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::spdzwise::ring {
class MulA1BTestAccess {
 public:
  static DaBit generate(SpdzWiseState* s, const Shape& shape) {
    return s->ring_->randomDaBit(shape);
  }
  static RepShare key(SpdzWiseState* s) { return s->ring_->key_; }
  static size_t pending(SpdzWiseState* s) { return s->ring_->pending_.size(); }
  static void corruptMac(SpdzWiseState* s, size_t party, bool last) {
    auto& record =
        last ? s->ring_->pending_.back() : s->ring_->pending_.front();
    if (s->rank() == party) record[2] += 1;
    if (s->rank() == (party + 2) % 3) record[3] += 1;
  }
  static void observe(SpdzWiseState* s, const std::shared_ptr<bool>& opened,
                      const std::shared_ptr<bool>& wire = nullptr) {
    auto original = s->boolean_->tag_;
    s->boolean_->tag_ = [original, opened, wire](std::string_view name) {
      if (name == "bit2a-open") *opened = true;
      if (wire) *wire = name == "bit2a-open";
      return original(name);
    };
  }
  static void fault(SpdzWiseState* s, NdArrayRef b, size_t party, int kind) {
    if (kind == 1) b = s->booleanBinary(b, b, true, false);
    auto r = generate(s, b.shape());
    if (kind == 0) corruptMac(s, party, false);
    if (kind == 1) {
      auto& t = s->boolean_->pending_.front();
      if (s->rank() == party) t[2][0] ^= 1;
      if (s->rank() == (party + 2) % 3) t[2][1] ^= 1;
    }
    if (kind == 2 && s->rank() == party)
      r.boolean.data<spdzwise::RepShare>()[0][0] ^= 1;
    try {
      s->finishBit2A(r, b);
    } catch (...) {
      s->abort();
      throw;
    }
  }
};
namespace {
RuntimeConfig Config() {
  RuntimeConfig c(SPDZWISE, FM64);
  c.spdzwise_config.domain = SpdzWiseConfig::RING;
  return c;
}
void Begin(SpdzWiseState* s) {
  s->begin(ExecutableProto("mula1b-test", {}, {"y"}, "test"), {});
}
NdArrayRef Plain(const Shape& shape, uint64_t value = 1) {
  NdArrayRef out(makeType<Pub2kTy>(FM64), shape);
  std::fill_n(out.data<uint64_t>(), out.numel(), value);
  return out;
}
NdArrayRef Bit(SpdzWiseState* s, const Shape& shape) {
  return s->publicBoolean(Plain(shape)).as(makeType<BShrTy>(s->session(), 1));
}
NdArrayRef Strided(const NdArrayRef& value) {
  NdArrayRef wide(value.eltype(), {value.numel(), 2});
  for (int64_t i = 0; i < value.numel(); ++i)
    std::memcpy(static_cast<char*>(wide.data()) + 2 * i * value.elsize(),
                static_cast<const char*>(value.data()) + i * value.elsize(),
                value.elsize());
  return wide.slice({0, 0}, {value.numel(), 1}, {1, 1}).reshape(value.shape());
}
struct Shares {
  NdArrayRef arithmetic, boolean, external;
  RepShare key;
};
void CheckAuthentication(const std::vector<Shares>& outputs, bool is_bit) {
  Word alpha = 0;
  for (const auto& r : outputs) alpha += r.key[0];
  for (int64_t i = 0; i < outputs[0].arithmetic.numel(); ++i) {
    Word a = 0, mac = 0;
    uint64_t b = 0;
    for (size_t p = 0; p < 3; ++p) {
      const auto& x = outputs[p].arithmetic.data<AuthShare>()[i];
      const auto& next = outputs[(p + 1) % 3].arithmetic.data<AuthShare>()[i];
      EXPECT_EQ(x[1], next[0]);
      EXPECT_EQ(x[3], next[2]);
      a += x[0];
      mac += x[2];
      if (is_bit) {
        const auto& y = outputs[p].boolean.data<spdzwise::RepShare>()[i];
        EXPECT_EQ(
            y[1],
            outputs[(p + 1) % 3].boolean.data<spdzwise::RepShare>()[i][0]);
        b ^= y[0];
      }
    }
    EXPECT_EQ(mac, alpha * a);
    if (is_bit) {
      EXPECT_TRUE(a == 0 || a == 1);
      EXPECT_EQ(uint64_t(a), b);
    }
  }
}
TEST(RingDaBit, DomainsMacFreshCallsAndTruncationInterleave) {
  for (int64_t n : {0, 1, 65, 129}) {
    const auto results = utils::simulate(
        3, [&](const std::shared_ptr<yacl::link::Context>& link) {
          SPUContext ctx(Config(), link);
          registerArithmeticPrototype(&ctx, link);
          auto* s = ctx.prot()->getState<SpdzWiseState>();
          ProtocolRunGuard guard(&ctx);
          Begin(s);
          std::vector<Shares> all;
          for (int call = 0; call < 3; ++call) {
            const auto bytes = link->GetStats()->sent_bytes.load();
            auto r = MulA1BTestAccess::generate(s, {n});
            if (!n) {
              EXPECT_EQ(bytes, link->GetStats()->sent_bytes.load());
            }
            EXPECT_EQ(r.arithmetic.shape(), Shape({n}));
            EXPECT_EQ(r.boolean.shape(), Shape({n}));
            EXPECT_EQ(r.boolean.eltype().as<BShrTy>()->nbits(), 1);
            EXPECT_EQ(r.boolean.eltype().as<BShrTy>()->session(), s->session());
            EXPECT_EQ(r.arithmetic.eltype().as<AShrTy>()->session(),
                      s->session());
            s->verify();
            EXPECT_EQ(MulA1BTestAccess::pending(s), 0);
            if (n && call) {
              EXPECT_NE(
                  std::memcmp(r.arithmetic.data(), all.back().arithmetic.data(),
                              n * sizeof(AuthShare)),
                  0);
            }
            all.push_back({r.arithmetic.clone(),
                           r.boolean.clone(),
                           {},
                           MulA1BTestAccess::key(s)});
            // Shared RandBit PRG/owner rotation must remain aligned with
            // truncation.
            auto truncated = s->truncPrRandomBit(s->publicValue(Plain({3}, 14)),
                                                 1, SignType::Unknown);
            s->verify();
            all.back().external = truncated.clone();
          }
          return all;
        });
    for (size_t call = 0; call < 3; ++call) {
      std::vector<Shares> sample;
      for (const auto& party : results) sample.push_back(party[call]);
      CheckAuthentication(sample, true);
      for (int64_t i = 0; i < 3; ++i) {
        Word truncated = 0;
        for (const auto& party : sample)
          truncated += party.external.data<AuthShare>()[i][0];
        EXPECT_EQ(uint64_t(truncated), 7);
      }
    }
  }
}
TEST(RingMulA1B, ValuesShapesStridesAndCheckedOutput) {
  for (const Shape& shape :
       {Shape{0}, Shape{}, Shape{65}, Shape{3, 43}, Shape{1025}}) {
    for (bool strided : {false, true}) {
      auto plain = Plain(shape), bits = Plain(shape);
      std::mt19937_64 rng(100);
      const std::array<uint64_t, 6> edge{
          0, 1, UINT64_MAX, uint64_t(1) << 63, (uint64_t(1) << 63) - 1, 65536};
      for (int64_t i = 0; i < plain.numel(); ++i) {
        plain.data<uint64_t>()[i] =
            size_t(i / 2) < edge.size() ? edge[i / 2] : rng();
        bits.data<uint64_t>()[i] = i & 1;
      }
      SpdzWiseIo io(FM64, 3, SpdzWiseConfig::RING);
      const auto aa = io.toShares(plain.as(makeType<RingTy>(FM64)), VIS_SECRET);
      const auto bb = io.toBooleanShares(bits.as(makeType<RingTy>(FM64)), 1);
      const auto results = utils::simulate(
          3, [&](const std::shared_ptr<yacl::link::Context>& link) {
            SPUContext ctx(Config(), link);
            registerArithmeticPrototype(&ctx, link);
            EXPECT_NE(dynamic_cast<MulA1B*>(ctx.prot()->getKernel("mul_a1b")),
                      nullptr);
            auto* s = ctx.prot()->getState<SpdzWiseState>();
            ProtocolRunGuard guard(&ctx);
            Value x(aa[link->Rank()], DT_U64), b(bb[link->Rank()], DT_I1);
            s->begin(ExecutableProto("mul-test", {"x", "b"}, {"y"}, "test"),
                     {x, b});
            x = import_s(&ctx, x);
            b = import_s(&ctx, b);
            if (strided && plain.numel()) {
              x = Value(Strided(x.data()), x.dtype());
              b = Value(Strided(b.data()), b.dtype());
              if (plain.numel() > 1) {
                EXPECT_FALSE(x.data().isCompact());
                EXPECT_FALSE(b.data().isCompact());
              }
            }
            const auto checks = s->arithmeticChecks();
            const auto before_bytes = link->GetStats()->sent_bytes.load();
            auto y = mul_a1b(&ctx, x, b);
            EXPECT_EQ(y.dtype(), x.dtype());
            EXPECT_EQ(y.shape(), shape);
            EXPECT_EQ(y.storage_type(), x.storage_type());
            EXPECT_EQ(s->arithmeticChecks(),
                      checks + size_t(plain.numel() != 0));
            EXPECT_EQ(MulA1BTestAccess::pending(s), size_t(plain.numel()));
            if (!plain.numel()) {
              EXPECT_EQ(before_bytes, link->GetStats()->sent_bytes.load());
            }
            s->checkMac();
            EXPECT_EQ(s->arithmeticChecks(),
                      checks + 2 * size_t(plain.numel() != 0));
            EXPECT_EQ(MulA1BTestAccess::pending(s), 0);
            const auto done = s->arithmeticChecks();
            auto internal = y.data().clone();
            s->prepareOutputs(&ctx, {"y"}, {y});
            guard.commit();
            EXPECT_EQ(done, s->arithmeticChecks());
            return Shares{internal,
                          {},
                          s->exportedValue("y").data(),
                          MulA1BTestAccess::key(s)};
          });
      CheckAuthentication(results, false);
      std::vector<NdArrayRef> ext;
      for (const auto& r : results) ext.push_back(r.external);
      auto actual = io.fromShares(ext);
      for (int64_t i = 0; i < plain.numel(); ++i)
        EXPECT_EQ(actual.data<uint64_t>()[i],
                  plain.data<uint64_t>()[i] * bits.data<uint64_t>()[i]);
    }
  }
}
TEST(RingMulA1B, InvalidArgumentsAbortAndLifecycle) {
  for (int kind = 0; kind < 5; ++kind) {
    utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* s = ctx.prot()->getState<SpdzWiseState>();
      ProtocolRunGuard guard(&ctx);
      Begin(s);
      auto x = s->publicValue(Plain({1}));
      auto b = Bit(s, {1});
      if (kind == 0) b = b.as(makeType<BShrTy>(s->session(), 2));
      if (kind == 1) b = Bit(s, {2});
      if (kind == 2) b = b.as(makeType<BShrTy>("other-session", 1));
      if (kind == 3) x = x.as(makeType<AShrTy>("other-session"));
      if (kind == 4) b.data<spdzwise::RepShare>()[0][0] = 2;
      EXPECT_ANY_THROW(
          dynDispatch(&ctx, "mul_a1b", Value(x, DT_U64), Value(b, DT_I1)));
      EXPECT_TRUE(s->failed());
      EXPECT_ANY_THROW(s->exportedValue("y"));
      EXPECT_ANY_THROW(s->mulA1B(x, b));
    });
  }
}
TEST(RingMulA1B, GenericDispatchAndOutputLifecycle) {
  for (bool reverse : {false, true}) {
    utils::simulate(3, [&](const auto& link) {
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* s = ctx.prot()->getState<SpdzWiseState>();
      ProtocolRunGuard guard(&ctx);
      Begin(s);
      Value x(s->publicValue(Plain({65}, UINT64_MAX)), DT_U64);
      Value b(Bit(s, {65}), DT_I1);
      auto y = reverse ? mul_ss(&ctx, b, x) : mul_ss(&ctx, x, b);
      EXPECT_EQ(y.storage_type(), x.storage_type());
      EXPECT_EQ(s->arithmeticChecks(), 1);
      EXPECT_EQ(MulA1BTestAccess::pending(s), 65);
      // The regular output transaction must discharge the product MACs.
      s->prepareOutputs(&ctx, {"y"}, {y});
      guard.commit();
      EXPECT_EQ(s->arithmeticChecks(), 2);
      EXPECT_EQ(MulA1BTestAccess::pending(s), 0);
      EXPECT_NO_THROW(s->exportedValue("y"));
      EXPECT_ANY_THROW(s->mulA1B(x.data(), b.data()));
    });
  }
}
TEST(RingMulA1B, PendingBooleanCheckedBeforeOpening) {
  utils::simulate(3, [&](const auto& link) {
    SPUContext ctx(Config(), link);
    registerArithmeticPrototype(&ctx, link);
    auto* s = ctx.prot()->getState<SpdzWiseState>();
    ProtocolRunGuard guard(&ctx);
    Begin(s);
    auto b = Bit(s, {65});
    b = s->booleanBinary(b, b, true, false);
    EXPECT_EQ(s->pendingBooleanBits(), 65);
    auto opened = std::make_shared<bool>(false);
    MulA1BTestAccess::observe(s, opened);
    auto y = s->mulA1B(s->publicValue(Plain({65}, 123)), b);
    EXPECT_TRUE(*opened);
    EXPECT_EQ(s->booleanChecks(), 1);
    EXPECT_EQ(s->pendingBooleanBits(), 0);
    s->checkMac();
    s->prepareOutputs(&ctx, {"y"}, {Value(y, DT_U64)});
    guard.commit();
  });
}
TEST(RingMulA1B, RejectsMaterialBooleanOpeningAndProductFaults) {
  for (size_t party = 0; party < 3; ++party)
    for (int kind = 0; kind < 4; ++kind) {
      utils::simulate(3, [&](const auto& link) {
        SPUContext ctx(Config(), link);
        registerArithmeticPrototype(&ctx, link);
        auto* s = ctx.prot()->getState<SpdzWiseState>();
        ProtocolRunGuard guard(&ctx);
        Begin(s);
        auto opened = std::make_shared<bool>(false);
        MulA1BTestAccess::observe(s, opened);
        auto b = Bit(s, {65});
        if (kind < 3) {
          EXPECT_ANY_THROW(MulA1BTestAccess::fault(s, b, party, kind));
          EXPECT_EQ(*opened, kind == 2);
        } else {
          auto y = s->mulA1B(s->publicValue(Plain({65}, 123)), b);
          MulA1BTestAccess::corruptMac(s, party, true);
          EXPECT_ANY_THROW(s->checkMac());
          EXPECT_ANY_THROW(s->prepareOutputs(&ctx, {"y"}, {Value(y, DT_U64)}));
        }
        EXPECT_TRUE(s->failed());
        EXPECT_ANY_THROW(s->exportedValue("y"));
      });
    }
}
class Bit2AFaultChannel : public yacl::link::transport::IChannel {
  std::shared_ptr<yacl::link::transport::IChannel> inner_;
  std::shared_ptr<bool> active_;
  yacl::Buffer change(yacl::Buffer value) {
    if (*active_ && value.size()) {
      value.data<char>()[0] ^= 1;
    }
    return value;
  }

 public:
  Bit2AFaultChannel(std::shared_ptr<yacl::link::transport::IChannel> inner,
                    std::shared_ptr<bool> active)
      : inner_(std::move(inner)), active_(std::move(active)) {}
  void SendAsync(const std::string& k, yacl::Buffer v) override {
    inner_->SendAsync(k, change(std::move(v)));
  }
  void SendAsyncThrottled(const std::string& k, yacl::Buffer v) override {
    inner_->SendAsyncThrottled(k, change(std::move(v)));
  }
  void Send(const std::string& k, yacl::ByteContainerView v) override {
    auto b = change(yacl::Buffer(v));
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
TEST(RingMulA1B, RejectsOpeningWireEquivocation) {
  for (size_t party = 0; party < 3; ++party) {
    utils::simulate(3, [&](const auto& base) {
      auto active = std::make_shared<bool>(false);
      auto opened = std::make_shared<bool>(false);
      yacl::link::ContextDesc desc;
      desc.id = base->Id();
      std::vector<std::shared_ptr<yacl::link::transport::IChannel>> channels(3);
      for (size_t i = 0; i < 3; ++i) {
        desc.parties.push_back({std::to_string(i), "memory"});
        if (i != base->Rank()) {
          channels[i] = base->GetChannel(i);
          if (base->Rank() == party && i == (party + 1) % 3)
            channels[i] =
                std::make_shared<Bit2AFaultChannel>(channels[i], active);
        }
      }
      auto link = std::make_shared<yacl::link::Context>(desc, base->Rank(),
                                                        channels, nullptr);
      link->SetRecvTimeout(5000);
      SPUContext ctx(Config(), link);
      registerArithmeticPrototype(&ctx, link);
      auto* s = ctx.prot()->getState<SpdzWiseState>();
      ProtocolRunGuard guard(&ctx);
      Begin(s);
      MulA1BTestAccess::observe(s, opened, active);
      EXPECT_ANY_THROW(
          s->mulA1B(s->publicValue(Plain({65}, 123)), Bit(s, {65})));
      EXPECT_TRUE(*opened);
      EXPECT_TRUE(s->failed());
      EXPECT_ANY_THROW(s->exportedValue("y"));
    });
  }
}
}  // namespace
}  // namespace spu::mpc::spdzwise::ring
