#include "minicpmo/audio_encoder.h"
#include "minicpmo/acl_context.h"
#include "minicpmo/ops.h"

#include <stdexcept>
#include <string>

namespace minicpmo {

AudioEncoderConfig default_minicpmo45_audio_config() {
    return AudioEncoderConfig{};
}

AudioEncoderWeights load_audio_encoder_weights(WeightsIndex& index, const AudioEncoderConfig& cfg) {
    AudioEncoderWeights w;
    // TODO: Load audio encoder weights from safetensors
    // Will implement once we have the actual model structure
    return w;
}

void encode_audio(const Tensor& mel_features,
                  const AudioEncoderWeights& w,
                  const AudioEncoderConfig& cfg,
                  Tensor& out,
                  aclrtStream stream) {
    // TODO: Implement audio encoding pipeline
    // Expected flow:
    // 1. Conv positional embedding on mel spectrogram
    // 2. Transformer encoder blocks (self-attention + FFN)
    // 3. Output hidden states for cross-modal fusion
    throw std::runtime_error("encode_audio not yet implemented");
}

}  // namespace minicpmo
