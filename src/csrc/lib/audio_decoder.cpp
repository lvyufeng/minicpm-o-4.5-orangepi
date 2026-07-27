#include "minicpmo/audio_decoder.h"
#include "minicpmo/acl_context.h"
#include "minicpmo/ops.h"

#include <stdexcept>
#include <string>

namespace minicpmo {

AudioDecoderConfig default_minicpmo45_audio_decoder_config() {
    return AudioDecoderConfig{};
}

AudioDecoderWeights load_audio_decoder_weights(WeightsIndex& index, const AudioDecoderConfig& cfg) {
    AudioDecoderWeights w;
    // TODO: Load audio decoder weights from safetensors
    return w;
}

void decode_audio(const Tensor& input_ids,
                  const Tensor& llm_hidden_states,
                  const AudioDecoderWeights& w,
                  const AudioDecoderConfig& cfg,
                  Tensor& logits,
                  aclrtStream stream) {
    // TODO: Implement audio decoding pipeline
    // Expected flow:
    // 1. Token embedding lookup
    // 2. Transformer decoder blocks (causal self-attn + cross-attn to LLM)
    // 3. Output logits over audio token vocab
    throw std::runtime_error("decode_audio not yet implemented");
}

}  // namespace minicpmo
