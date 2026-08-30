#include "llmetal/cpu/gqa.hpp"
#include "llmetal/cpu/softmax.hpp"
#include <string>

namespace llmetal::cpu {

// alias for std::uint32_t
using uint = std::uint32_t;
// alias for -inf
constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

void gqaPrefillFlash(
    const CpuTensor<float>& q, // [B, S, NQ (num query heads), D (hidden dim per head)]
    const CpuTensor<float>& k, // [B, S, NKV, D]
    const CpuTensor<float>& v, // [B, S, NKV, D]
    CpuTensor<float>& output   // [B, S, NQ, D]
) {

    if (q.shape().rank() != 4) throw std::invalid_argument("Invalid q shape rank, got " + std::to_string(q.shape().rank()));
    if (k.shape().rank() != 4) throw std::invalid_argument("Invalid k shape rank, got " + std::to_string(k.shape().rank()));
    if (v.shape().rank() != 4) throw std::invalid_argument("Invalid v shape rank, got " + std::to_string(v.shape().rank()));
    if (output.shape().rank() != 4) throw std::invalid_argument("Invalid output shape rank, got " + std::to_string(output.shape().rank()));

    const uint B = checked_u32(q.shape()[0], "gqa_prefill_B");
    const uint S = checked_u32(q.shape()[1], "gqa_prefill_S");
    const uint NQ = checked_u32(q.shape()[2], "gqa_prefill_NQ");
    const uint D = checked_u32(q.shape()[3], "gqa_prefill_D");
    const uint NKV = checked_u32(k.shape()[2], "gqa_prefill_NKV");

    if (k.shape()[0] != B) throw std::invalid_argument("Invalid k shape[0] - expected B (" + std::to_string(B) + "), got " + std::to_string(k.shape()[0]));
    if (k.shape()[1] != S) throw std::invalid_argument("Invalid k shape[1] - expected S (" + std::to_string(S) + "), got " + std::to_string(k.shape()[1]));
    if (k.shape()[3] != D) throw std::invalid_argument("Invalid k shape[3] - expected D (" + std::to_string(D) + "), got " + std::to_string(k.shape()[3]));
    if (v.shape()[0] != B) throw std::invalid_argument("Invalid v shape[0] - expected B (" + std::to_string(B) + "), got " + std::to_string(v.shape()[0]));
    if (v.shape()[1] != S) throw std::invalid_argument("Invalid v shape[1] - expected S (" + std::to_string(S) + "), got " + std::to_string(v.shape()[1]));
    if (v.shape()[3] != D) throw std::invalid_argument("Invalid v shape[3] - expected D (" + std::to_string(D) + "), got " + std::to_string(v.shape()[3]));
    if (v.shape()[2] != NKV) throw std::invalid_argument("Invalid v shape[2] - expected k shape[2] == NKV (" + std::to_string(NKV) + "), got " + std::to_string(v.shape()[2]));
    if (output.shape()[0] != B) throw std::invalid_argument("Invalid output shape[0] - expected B (" + std::to_string(B) + "), got " + std::to_string(output.shape()[0]));
    if (output.shape()[1] != S) throw std::invalid_argument("Invalid output shape[1] - expected S (" + std::to_string(S) + "), got " + std::to_string(output.shape()[1]));
    if (output.shape()[2] != NQ) throw std::invalid_argument("Invalid output shape[2] - expected NQ (" + std::to_string(NQ) + "), got " + std::to_string(output.shape()[2]));
    if (output.shape()[3] != D) throw std::invalid_argument("Invalid output shape[3] - expected D (" + std::to_string(D) + "), got " + std::to_string(output.shape()[3]));
    if (NQ % NKV != 0) throw std::invalid_argument("Invalid NQ (" + std::to_string(NQ) + ") - must be a multiple of NKV (" + std::to_string(NKV) + ")");

    const uint repeats = NQ / NKV;
    const float scale = std::powf(D, -0.5);
    // Per-query-token D row accumulator
    std::vector<float> acc(D);
    for (uint b = 0; b < B; ++b) {
        for (uint qh = 0; qh < NQ; ++qh) {
            uint kvh = qh / repeats;

            // Flash attention per query token
            for (uint qs = 0; qs < S; ++qs) {
                uint q_offset = ((b * S + qs) * NQ + qh) * D;

                // Keep running max and sum of exp, and zero the row accumulator
                float max_row = -std::numeric_limits<float>::infinity();
                float sum_row = 0.0f;
                for (uint d = 0; d < D; ++d) acc[d] = 0.0f;

                // Causal dot for query with every key
                for (uint ks = 0; ks <= qs; ++ks) {
                    uint kv_offset = ((b * S + ks) * NKV + kvh) * D;
                    
                    float curr_scale = 0.0f;
                    for (uint d = 0; d < D; ++d)
                        curr_scale += q[q_offset + d] * k[kv_offset + d];
                    curr_scale *= scale;

                    // Compute the update factors
                    float max_curr = std::max(max_row, curr_scale);
                    // Correction factor
                    float alpha = max_row != NEG_INF ? std::exp(max_row - max_curr) : 0.0f;
                    // Current value token factor
                    float beta = std::exp(curr_scale - max_curr);

                    // Update accumulators
                    sum_row = sum_row * alpha + beta;
                    max_row = max_curr;

                    // Update accumulator
                    for (uint d = 0; d < D; ++d) {
                        acc[d] = acc[d] * alpha + v[kv_offset + d] * beta;
                    }
                }

                // Final scaling
                float inv_sum = 1.0f / sum_row;
                for (uint d = 0; d < D; ++d) output[q_offset + d] = inv_sum * acc[d];
            }
        }
    }
}

void gqaPrefill(
    const CpuTensor<float>& q, // [B, S, NQ (num query heads), D (hidden dim per head)]
    const CpuTensor<float>& k, // [B, S, NKV, D]
    const CpuTensor<float>& v, // [B, S, NKV, D]
    CpuTensor<float>& output   // [B, S, NQ, D]
) {

    if (q.shape().rank() != 4) throw std::invalid_argument("Invalid q shape rank, got " + std::to_string(q.shape().rank()));
    if (k.shape().rank() != 4) throw std::invalid_argument("Invalid k shape rank, got " + std::to_string(k.shape().rank()));
    if (v.shape().rank() != 4) throw std::invalid_argument("Invalid v shape rank, got " + std::to_string(v.shape().rank()));
    if (output.shape().rank() != 4) throw std::invalid_argument("Invalid output shape rank, got " + std::to_string(output.shape().rank()));

    uint B = checked_u32(q.shape()[0], "gqa_prefill_B");
    uint S = checked_u32(q.shape()[1], "gqa_prefill_S");
    uint NQ = checked_u32(q.shape()[2], "gqa_prefill_NQ");
    uint D = checked_u32(q.shape()[3], "gqa_prefill_D");
    uint NKV = checked_u32(k.shape()[2], "gqa_prefill_NKV");

    if (k.shape()[0] != B) throw std::invalid_argument("Invalid k shape[0] - expected B (" + std::to_string(B) + "), got " + std::to_string(k.shape()[0]));
    if (k.shape()[1] != S) throw std::invalid_argument("Invalid k shape[1] - expected S (" + std::to_string(S) + "), got " + std::to_string(k.shape()[1]));
    if (k.shape()[3] != D) throw std::invalid_argument("Invalid k shape[3] - expected D (" + std::to_string(D) + "), got " + std::to_string(k.shape()[3]));
    if (v.shape()[0] != B) throw std::invalid_argument("Invalid v shape[0] - expected B (" + std::to_string(B) + "), got " + std::to_string(v.shape()[0]));
    if (v.shape()[1] != S) throw std::invalid_argument("Invalid v shape[1] - expected S (" + std::to_string(S) + "), got " + std::to_string(v.shape()[1]));
    if (v.shape()[3] != D) throw std::invalid_argument("Invalid v shape[3] - expected D (" + std::to_string(D) + "), got " + std::to_string(v.shape()[3]));
    if (v.shape()[2] != NKV) throw std::invalid_argument("Invalid v shape[2] - expected k shape[2] == NKV (" + std::to_string(NKV) + "), got " + std::to_string(v.shape()[2]));
    if (output.shape()[0] != B) throw std::invalid_argument("Invalid output shape[0] - expected B (" + std::to_string(B) + "), got " + std::to_string(output.shape()[0]));
    if (output.shape()[1] != S) throw std::invalid_argument("Invalid output shape[1] - expected S (" + std::to_string(S) + "), got " + std::to_string(output.shape()[1]));
    if (output.shape()[2] != NQ) throw std::invalid_argument("Invalid output shape[2] - expected NQ (" + std::to_string(NQ) + "), got " + std::to_string(output.shape()[2]));
    if (output.shape()[3] != D) throw std::invalid_argument("Invalid output shape[3] - expected D (" + std::to_string(D) + "), got " + std::to_string(output.shape()[3]));
    if (NQ % NKV != 0) throw std::invalid_argument("Invalid NQ (" + std::to_string(NQ) + ") - must be a multiple of NKV (" + std::to_string(NKV) + ")");

    uint repeats = NQ / NKV;
    float scale = std::powf(D, -0.5);

    // For non-flash-attention
    CpuTensor<uint> valid({ S }); for (uint i = 0; i < S; ++i) valid[i] = i + 1;
    CpuTensor<float> score_matrix({S, S});
    for (uint b = 0; b < B; ++b) {
        for (uint qh = 0; qh < NQ; ++qh) {
            uint kvh = qh / repeats;

            // For this query head and key head, compute the attention - dot each query token with each key
            for (uint qs = 0; qs < S; ++qs) {
                uint q_offset = ((b * S + qs) * NQ + qh) * D;
                for (uint ks = 0; ks < S; ++ks) {
                    uint k_offset = ((b * S + ks) * NKV + kvh) * D;
                    
                    float dot = 0.0f;
                    for (uint d = 0; d < D; ++d) {
                        dot += q[q_offset + d] * k[k_offset + d];
                    }
                    dot *= scale;
                    score_matrix[qs * S + ks] = dot;
                }
            }
            
            // Softmax
            llmetal::cpu::softmax(score_matrix, valid, score_matrix);

            // Scale V values
            for (uint os = 0; os < S; ++os) {
                uint o_offset = ((b * S + os) * NQ + qh) * D;
                // Set to 0 first, in case of reused buffer
                for (uint d = 0; d < D; ++d) output[o_offset + d] = 0; 

                for (uint vs = 0; vs <= os; ++vs) {
                    uint v_offset = ((b * S + vs) * NKV + kvh) * D;
                    float score_value = score_matrix[os * S + vs];
                    
                    for (uint d = 0; d < D; ++d) { 
                        output[o_offset + d] += score_value * v[v_offset + d];
                    }
                }
            }
        }
    }
}


} // namespace llmetal::cpu
