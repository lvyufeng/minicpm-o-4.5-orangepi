#pragma once

#include "minicpmo/tensor.h"
#include "minicpmo/weights.h"

#include <acl/acl.h>
#include <cstdint>
#include <vector>

namespace minicpmo {

struct AudioEncoderConfig {
    int64_t hidden_size{768};
    int64_t num_mel_bins{128};
    int64_t num_hidden_layers{12};
    int64_t num_attention_heads{12};
    int64_t intermediate_size{3072};
    double layer_norm_eps{1e-6};
};

AudioEncoderConfig default_minicpmo45_audio_config();

struct AudioEncoderWeights {
    // TODO: Define audio encoder weight tensors
    // Based on Whisper-like architecture expected in MiniCPM-O
};

AudioEncoderWeights load_audio_encoder_weights(WeightsIndex& index, const AudioEncoderConfig& cfg);

// Encode audio mel-spectrogram features to hidden representations
void encode_audio(const Tensor& mel_features,
                  const AudioEncoderWeights& w,
                  const AudioEncoderConfig& cfg,
                  Tensor& out,
                  aclrtStream stream);

}  // namespace minicpmo
