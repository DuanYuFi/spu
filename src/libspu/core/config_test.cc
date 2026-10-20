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

#include "libspu/core/config.h"

#include "gtest/gtest.h"

#include "libspu/spu.pb.h"

namespace spu {
namespace {

TEST(RuntimeConfig, SpdzWiseDefaultsToFieldForLegacyConfigurations) {
  const auto config = makeFullRuntimeConfig(RuntimeConfig(SPDZWISE, FM64));
  EXPECT_EQ(config.spdzwise_config.domain, SpdzWiseConfig::FIELD);
  EXPECT_EQ(config.fxp_fraction_bits, 16);
  RuntimeConfig parsed;
  ASSERT_TRUE(
      parsed.ParseFromJsonString(R"({"protocol":"SPDZWISE","field":"FM64"})"));
  EXPECT_EQ(parsed.spdzwise_config.domain, SpdzWiseConfig::FIELD);
  EXPECT_NO_THROW(populateRuntimeConfig(parsed));
}

TEST(RuntimeConfig, SpdzWiseDomainSurvivesProtoBinaryAndJson) {
  for (auto domain : {SpdzWiseConfig::FIELD, SpdzWiseConfig::RING}) {
    RuntimeConfig config(SPDZWISE, FM64);
    config.spdzwise_config.domain = domain;
    EXPECT_NO_THROW(populateRuntimeConfig(config));
    const auto proto = config.ToProto();
    EXPECT_EQ(static_cast<int>(proto.spdzwise_config().domain()), domain);
    EXPECT_EQ(RuntimeConfig(proto).spdzwise_config.domain, domain);
    RuntimeConfig parsed;
    ASSERT_TRUE(parsed.ParseFromString(config.SerializeAsString()));
    EXPECT_EQ(parsed.spdzwise_config.domain, domain);
    const std::string name = domain == SpdzWiseConfig::FIELD ? "FIELD" : "RING";
    ASSERT_TRUE(parsed.ParseFromJsonString(
        R"({"protocol":"SPDZWISE","field":"FM64","spdzwise_config":{"domain":")" +
        name + R"("}})"));
    EXPECT_EQ(parsed.spdzwise_config.domain, domain);
    EXPECT_NE(parsed.DumpToString().find("domain: " + name), std::string::npos);
  }
}

TEST(RuntimeConfig, SpdzWiseRejectsUnsupportedStorageAndDomains) {
  for (auto domain : {SpdzWiseConfig::FIELD, SpdzWiseConfig::RING}) {
    for (auto storage : {FM32, FM128}) {
      RuntimeConfig config(SPDZWISE, storage);
      config.spdzwise_config.domain = domain;
      EXPECT_ANY_THROW(populateRuntimeConfig(config));
    }
  }
  RuntimeConfig config(SPDZWISE, FM64);
  config.spdzwise_config.domain = static_cast<SpdzWiseConfig::Domain>(99);
  EXPECT_ANY_THROW(populateRuntimeConfig(config));
}

TEST(RuntimeConfig, OtherProtocolsRetainTheirStorageChoices) {
  for (auto storage : {FM32, FM64, FM128}) {
    RuntimeConfig config(ABY3, storage);
    EXPECT_NO_THROW(populateRuntimeConfig(config));
  }
}

}  // namespace
}  // namespace spu
