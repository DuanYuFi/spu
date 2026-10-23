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

#include <functional>
#include <memory>

#include "yacl/base/buffer.h"

#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/spdzwise/type.h"

namespace spu::mpc::spdzwise {
namespace ring {
class ConversionTestAccess;
class MulA1BTestAccess;
}

// Counts packed 64-bit triples (including one zero-extended partial word),
// not the number of active bit lanes. Callers pad nonempty checking batches
// to at least 1024 words before selecting a bucket size.
size_t AbflBucketSize(size_t words);

// Owned by the checked session. No application reveal API is exposed.
class BooleanState {
  friend class BooleanTestAccess;
  friend class SpdzWiseState;
  friend class ring::ConversionTestAccess;
  friend class ring::MulA1BTestAccess;
  using Triple = std::array<RepShare, 3>;
  // Default-initialized storage: parallel writers touch each element once.
  // Unlike vector(size), this does not zero large buffers on the caller thread.
  struct TriplePool {
    size_t count;
    std::unique_ptr<Triple[]> storage;
    explicit TriplePool(size_t n) : count(n), storage(new Triple[n]) {}
    size_t size() const { return count; }
    Triple& operator[](size_t i) { return storage[i]; }
    const Triple& operator[](size_t i) const { return storage[i]; }
  };
  std::shared_ptr<yacl::link::Context> link_;
  std::function<std::string(std::string_view)> tag_;
  PrgState random_prg_, mask_prg_;
  std::vector<Triple> pending_;  // full words packed from narrow inputs
  // Full-width calls own contiguous snapshots; adding a call never relocates
  // the records of previous calls. Verification traverses blocks directly.
  std::vector<std::vector<Triple>> pending_batches_;
  Triple overflow_{};
  size_t overflow_bits_ = 0;
  size_t bits_ = 0;
  std::string layout_;
  uint64_t checks_ = 0;

  // With borrow_local, the local result slot is empty; the caller keeps bytes
  // alive and reads its local view directly. Peer slots always own their data.
  std::vector<yacl::Buffer> gather(std::string_view bytes,
                                   std::string_view name,
                                   bool borrow_local = false);
  void agree(std::string_view bytes);
  void requireAll(bool valid, std::string_view reason);
  std::vector<RepShare> andRaw(const std::vector<RepShare>& x,
                               const std::vector<RepShare>& y);
  std::vector<uint64_t> open(absl::Span<const RepShare> shares,
                             std::string_view operation = "boolean-open");
  void open(absl::Span<const RepShare> shares, absl::Span<uint64_t> result,
            std::string_view operation = "boolean-open");
  void reservePending(size_t full_words);
  void append(const std::vector<RepShare>& a, const std::vector<RepShare>& b,
              const std::vector<RepShare>& c, size_t width);
  TriplePool generate(size_t words);
  size_t checkWords() const;
  void checkPrepared(TriplePool pool);
  std::vector<size_t> permutation(size_t count);
  static std::vector<size_t> shuffledIndices(size_t count, uint128_t seed);

 public:
  BooleanState(std::shared_ptr<yacl::link::Context> link,
               std::function<std::string(std::string_view)> tag);
  size_t pendingBits() const { return bits_; }
  uint64_t checks() const { return checks_; }
  void checkCopies(const std::vector<RepShare>& shares);
  NdArrayRef andBB(const NdArrayRef& x, const NdArrayRef& y,
                   const std::string& session);
  void check();
};

// Shared ABFL kernel registration, independent of the arithmetic domain.
void registerBooleanKernels(Object* object);

}  // namespace spu::mpc::spdzwise
