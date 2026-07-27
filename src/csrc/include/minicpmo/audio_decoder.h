#pragma once

#include "minicpmo/tensor.h"
#include "minicpmo/weights.h"

#include <acl/acl.h>
#include <cstdint>
#include <vector>

namespace minicpmo {

struct AudioDecoderConfig {
    int64_t hidden_size{768};
    int64_t num_hidden_layers{12};
    int64_t num_attention_heads{12};
    int64_t intermediate_size{3072};
    int64_t vocab_size{51865};  // Whisper tokenizer vocab size
    double layer_norm_eps{1e-6};
};

AudioDecoderConfig default_minicpmo45_audio_decoder_config();

struct AudioDecoderWeights {
    // TODO: Define audio decoder weight tensors
    // Cross-attention to LLM hidden states + causal self-attention
};

AudioDecoderWeights load_audio_decoder_weights(WeightsIndex& index, const AudioDecoderConfig& cfg);

// Decode audio tokens (for audio generation tasks)
void decode_audio(const Tensor& input_ids,
                  const Tensor& llm_hidden_states,
                  const AudioDecoderWeights& w,
                  const AudioDecoderConfig& cfg,
                  Tensor& logits,
                  aclrtStream stream);

}  // namespace minicpmo
