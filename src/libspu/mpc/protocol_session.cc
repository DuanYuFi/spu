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

#include "libspu/mpc/protocol_session.h"

#include "libspu/mpc/api.h"

namespace spu::mpc {

void ProtocolSession::requireHealthy() const {
  SPU_ENFORCE(!failed_, "protocol session failed; create a new runtime");
}

void ProtocolSession::requireRunning() const {
  requireHealthy();
  SPU_ENFORCE(running_, "operation requires an active protocol execution");
}

void ProtocolSession::begin(const ExecutableProto& exec,
                            const std::vector<Value>& inputs) {
  requireHealthy();
  SPU_ENFORCE(!running_, "overlapping executions are not supported");
  running_ = true;
  prepared_ = false;
  pending_exports_.clear();
  committed_exports_.clear();
  try {
    onBegin(exec, inputs);
  } catch (...) {
    abort();
    throw;
  }
}

void ProtocolSession::prepareOutputs(SPUContext* ctx,
                                     const std::vector<std::string>& names,
                                     const std::vector<Value>& outputs) {
  requireRunning();
  try {
    SPU_ENFORCE(!prepared_ && names.size() == outputs.size(),
                "invalid protocol output transaction");
    // Mandatory for a session: silently skipping this hook would release
    // unchecked outputs. Ordinary protocols never enter this path.
    KernelEvalContext finalization(ctx);
    ctx->prot()->getKernel("finalize_protocol")->evaluate(&finalization);
    std::map<std::string, Value> exports;
    for (size_t i = 0; i < outputs.size(); ++i) {
      SPU_ENFORCE(!outputs[i].isComplex(),
                  "complex checked output unsupported");
      const auto value =
          outputs[i].isSecret()
              ? export_s(ctx, outputs[i], exportType(names[i], outputs[i]))
              : outputs[i];
      SPU_ENFORCE(value.storage_type().isSerializable(),
                  "export returned a protocol-private value");
      SPU_ENFORCE(exports.emplace(names[i], value).second,
                  "duplicate output name");
    }
    pending_exports_ = std::move(exports);
    prepared_ = true;
  } catch (...) {
    abort();
    throw;
  }
}

void ProtocolSession::commit() {
  requireRunning();
  SPU_ENFORCE(prepared_, "protocol outputs were not checked");
  committed_exports_.swap(pending_exports_);
  pending_exports_.clear();
  running_ = false;
  prepared_ = false;
}

void ProtocolSession::abort() noexcept {
  failed_ = true;
  running_ = false;
  prepared_ = false;
  pending_exports_.clear();
  committed_exports_.clear();
  onAbort();
}

void ProtocolSession::invalidateExport(const std::string& name) {
  requireHealthy();
  SPU_ENFORCE(!running_, "cannot replace inputs during an execution");
  committed_exports_.erase(name);
}

void ProtocolSession::clearExports() {
  requireHealthy();
  SPU_ENFORCE(!running_, "cannot clear inputs during an execution");
  committed_exports_.clear();
}

const Value& ProtocolSession::exportedValue(const std::string& name) const {
  requireHealthy();
  SPU_ENFORCE(!running_, "outputs are unavailable during execution");
  const auto it = committed_exports_.find(name);
  SPU_ENFORCE(it != committed_exports_.end(), "no checked export for {}", name);
  return it->second;
}

ProtocolSession* getProtocolSession(SPUContext* ctx) {
  return ctx->prot()->hasState(ProtocolSession::kBindName())
             ? ctx->prot()->getState<ProtocolSession>()
             : nullptr;
}

}  // namespace spu::mpc
