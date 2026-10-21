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

#pragma once

#include "libspu/core/config.h"
#include "libspu/mpc/io_interface.h"

namespace spu::mpc::spdzwise {

// Domain-specific arithmetic IO and shared ABFL Boolean IO.
class SpdzWiseIo final : public BaseIo {
  SpdzWiseConfig::Domain domain_;

 public:
  SpdzWiseIo(FieldType field, size_t parties,
             SpdzWiseConfig::Domain domain = SpdzWiseConfig::FIELD);
  Type getShareType(Visibility vis, int owner_rank = -1) const override;
  std::vector<NdArrayRef> toShares(const NdArrayRef& raw, Visibility vis,
                                   int owner_rank = -1) const override;
  std::vector<NdArrayRef> toBooleanShares(const NdArrayRef& raw,
                                          size_t nbits) const;
  std::vector<NdArrayRef> makeBitSecret(const PtBufferView& raw) const override;
  size_t getBitSecretShareSize(size_t numel) const override;
  bool hasBitSecretSupport() const override { return true; }
  NdArrayRef fromShares(const std::vector<NdArrayRef>& shares) const override;
};

std::unique_ptr<IoInterface> makeSpdzWiseIo(const RuntimeConfig& config,
                                            size_t parties);

}  // namespace spu::mpc::spdzwise
