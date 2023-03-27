
#include "libspu/mpc/beaver/state.h"

namespace spu::mpc {

void gen_arith_triples(Object* ctx,
                       std::vector<beaver::ArithmeticTriple>* triples,
                       size_t num_triples,
                       size_t batch_size = BeaverState::batch_size_,
                       size_t bucket_size = BeaverState::bucket_size_);

void gen_bin_triples(Object* ctx, std::vector<beaver::BinaryTriple>* triples,
                     size_t num_triples,
                     size_t batch_size = BeaverState::batch_size_,
                     size_t bucket_size = BeaverState::bucket_size_);

std::vector<beaver::ArithmeticTriple> cut_and_choose(
    Object* ctx, typename std::vector<beaver::ArithmeticTriple>::iterator data,
    size_t batch_size, size_t bucket_size, size_t C);

std::vector<beaver::BinaryTriple> cut_and_choose(
    Object* ctx, typename std::vector<beaver::BinaryTriple>::iterator data,
    size_t batch_size, size_t bucket_size, size_t C);

}  // namespace spu::mpc
