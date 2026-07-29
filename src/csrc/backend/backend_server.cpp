// Backend TCP server implementing JSON protocol for MiniCPM-o-Demo integration
//
// Protocol: each message is a JSON object terminated by newline
// Request format:
//   {
//     "method": "load_model" | "chat_prefill" | "chat_generate" | "metrics" | "shutdown",
//     "params": { method-specific parameters }
//   }
//
// Response format:
//   {
//     "status": "ok" | "error",
//     "result": { method-specific result },
//     "error": "error message if status==error"
//   }

#include "minicpmo/acl_context.h"
#include "minicpmo/audio_decoder.h"
#include "minicpmo/audio_encoder.h"
#include "minicpmo/decoder_layer.h"
#include "minicpmo/language_model.h"
#include "minicpmo/ops.h"
#include "minicpmo/profiling.h"
#include "minicpmo/vision.h"
#include "minicpmo/weights.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Simple JSON parser/builder (minimal implementation for protocol needs)
namespace json {

struct Value {
    enum Type { Object, Array, String, Number, Boolean, Null } type;
    std::string str_val;
    double num_val;
    bool bool_val;
    std::map<std::string, Value> obj_val;
    std::vector<Value> arr_val;

    Value() : type(Null) {}
    explicit Value(const std::string& s) : type(String), str_val(s) {}
    explicit Value(const char* s) : type(String), str_val(s) {}
    explicit Value(double n) : type(Number), num_val(n) {}
    explicit Value(bool b) : type(Boolean), bool_val(b) {}

    static Value object() { Value v; v.type = Object; return v; }
    static Value array() { Value v; v.type = Array; return v; }

    Value& operator[](const std::string& key) {
        if (type != Object) type = Object;
        return obj_val[key];
    }
    const Value& at(const std::string& key) const { return obj_val.at(key); }
    bool contains(const std::string& key) const { return obj_val.count(key) > 0; }

    std::string as_string() const { return str_val; }
    int as_int() const { return static_cast<int>(num_val); }
    double as_double() const { return num_val; }
    bool as_bool() const { return bool_val; }

    std::vector<int32_t> as_int_array() const {
        std::vector<int32_t> result;
        if (type == Array) {
            for (const auto& v : arr_val) {
                result.push_back(v.as_int());
            }
        }
        return result;
    }
};

// Minimal JSON serializer
std::string dumps(const Value& v) {
    std::ostringstream oss;
    if (v.type == Value::Object) {
        oss << "{";
        bool first = true;
        for (const auto& kv : v.obj_val) {
            if (!first) oss << ",";
            first = false;
            oss << "\"" << kv.first << "\":";
            oss << dumps(kv.second);
        }
        oss << "}";
    } else if (v.type == Value::Array) {
        oss << "[";
        bool first = true;
        for (const auto& elem : v.arr_val) {
            if (!first) oss << ",";
            first = false;
            oss << dumps(elem);
        }
        oss << "]";
    } else if (v.type == Value::String) {
        oss << "\"" << v.str_val << "\"";
    } else if (v.type == Value::Number) {
        oss << v.num_val;
    } else if (v.type == Value::Boolean) {
        oss << (v.bool_val ? "true" : "false");
    } else {
        oss << "null";
    }
    return oss.str();
}

// Minimal JSON parser (only handles well-formed input from our Python client)
Value loads(const std::string& text) {
    // Simplified parser - in production use a proper JSON library
    // Parse top-level string fields: "type", "model_path", "session_id", etc.
    Value root = Value::object();

    // Helper to extract string value for a given key
    auto extract_string = [&](const std::string& key) -> bool {
        std::string pattern = "\"" + key + "\"";
        size_t pos = text.find(pattern);
        if (pos != std::string::npos) {
            size_t start = text.find("\"", pos + pattern.length() + 1);
            if (start != std::string::npos) {
                size_t end = text.find("\"", start + 1);
                if (end != std::string::npos) {
                    root[key] = Value(text.substr(start + 1, end - start - 1));
                    return true;
                }
            }
        }
        return false;
    };

    // Helper to extract integer array for a given key
    auto extract_int_array = [&](const std::string& key) -> bool {
        std::string pattern = "\"" + key + "\"";
        size_t pos = text.find(pattern);
        if (pos != std::string::npos) {
            size_t arr_start = text.find("[", pos);
            if (arr_start != std::string::npos) {
                size_t arr_end = text.find("]", arr_start);
                if (arr_end != std::string::npos) {
                    Value arr = Value::array();
                    std::string arr_content = text.substr(arr_start + 1, arr_end - arr_start - 1);

                    // Simple comma-split parser
                    std::istringstream iss(arr_content);
                    std::string token;
                    while (std::getline(iss, token, ',')) {
                        // Trim whitespace
                        size_t first = token.find_first_not_of(" \t\n\r");
                        size_t last = token.find_last_not_of(" \t\n\r");
                        if (first != std::string::npos && last != std::string::npos) {
                            token = token.substr(first, last - first + 1);
                            if (!token.empty()) {
                                arr.arr_val.push_back(Value(std::stod(token)));
                            }
                        }
                    }
                    root[key] = arr;
                    return true;
                }
            }
        }
        return false;
    };

    // Helper to extract numeric value for a given key
    auto extract_number = [&](const std::string& key) -> bool {
        std::string pattern = "\"" + key + "\":";
        size_t pos = text.find(pattern);
        if (pos != std::string::npos) {
            size_t start = pos + pattern.length();
            // Skip whitespace
            while (start < text.length() && std::isspace(text[start])) ++start;
            // Extract number
            size_t end = start;
            while (end < text.length() && (std::isdigit(text[end]) || text[end] == '.' || text[end] == '-')) {
                ++end;
            }
            if (end > start) {
                std::string num_str = text.substr(start, end - start);
                root[key] = Value(std::stod(num_str));
                return true;
            }
        }
        return false;
    };

    // Extract common fields
    extract_string("type");
    extract_string("model_path");
    extract_string("session_id");
    extract_string("method");
    extract_int_array("input_ids");
    extract_number("max_new_tokens");
    extract_number("length_penalty");
    extract_number("request_id");

    // For backward compatibility, also check "method" field
    if (!root.contains("type") && root.contains("method")) {
        root["type"] = root["method"];
    }

    return root;
}

}  // namespace json

using namespace minicpmo;

class BackendSession {
public:
    BackendSession(int device_id) : device_id_(device_id) {
        ctx_ = std::make_unique<AclContext>(device_id);
    }

    void load_model(const std::string& model_path) {
        std::cout << "[Session] Loading model from: " << model_path << std::endl;

        // Load weights
        weights_index_ = std::make_unique<WeightsIndex>(model_path + "/model.safetensors");

        // Load configs
        lm_cfg_ = default_minicpmo45_lm_config();
        vision_cfg_ = default_minicpmo45_vision_config();

        // Load model weights
        lm_weights_ = load_language_model_weights(*weights_index_, lm_cfg_);
        vision_weights_ = load_vision_weights(*weights_index_, vision_cfg_);

        // Release shard file memory (unified memory on Orange Pi 310B)
        weights_index_->release_shard_memory();

        // Build RoPE tables
        const int64_t max_seq = 4096;
        cos_table_ = Tensor({max_seq, lm_cfg_.rotary_dim / 2}, DType::Float16);
        sin_table_ = Tensor({max_seq, lm_cfg_.rotary_dim / 2}, DType::Float16);
        build_rope_tables(max_seq, lm_cfg_, cos_table_, sin_table_);

        // Initialize decode state
        std::vector<std::string> layer_types;
        for (int i = 0; i < lm_cfg_.num_layers; i++) {
            layer_types.push_back(lm_cfg_.layer_types[i]);
        }

        FullAttentionDecoderLayerConfig full_cfg;
        full_cfg.num_q_heads = lm_cfg_.num_q_heads;
        full_cfg.num_kv_heads = lm_cfg_.num_kv_heads;
        full_cfg.head_dim = lm_cfg_.head_dim;
        full_cfg.rotary_dim = lm_cfg_.rotary_dim;
        full_cfg.rms_epsilon = lm_cfg_.rms_epsilon;

        decode_state_ = std::make_unique<DecodeState>(
            make_decode_state(max_seq, layer_types, full_cfg, ctx_->stream()));

        std::cout << "[Session] Model loaded successfully" << std::endl;

        // Warmup: compile all kernels by running a dummy inference
        // Warmup disabled - causes timeout due to JIT compilation
        std::cout << "[Session] Skipping warmup (first inference will be slow due to JIT)" << std::endl;

        model_loaded_ = true;
    }

    json::Value chat_prefill(const json::Value& params) {
        if (!model_loaded_) {
            throw std::runtime_error("Model not loaded");
        }

        // Reset decode state so a new conversation turn can be prefilled.
        // This is safe to call repeatedly — each prefill starts a fresh context.
        std::vector<std::string> layer_types;
        for (int i = 0; i < lm_cfg_.num_layers; i++) layer_types.push_back(lm_cfg_.layer_types[i]);
        FullAttentionDecoderLayerConfig full_cfg;
        full_cfg.num_q_heads  = lm_cfg_.num_q_heads;
        full_cfg.num_kv_heads = lm_cfg_.num_kv_heads;
        full_cfg.head_dim     = lm_cfg_.head_dim;
        full_cfg.rotary_dim   = lm_cfg_.rotary_dim;
        full_cfg.rms_epsilon  = lm_cfg_.rms_epsilon;
        decode_state_ = std::make_unique<DecodeState>(
            make_decode_state(4096, layer_types, full_cfg, ctx_->stream()));
        prefill_done_ = false;

        // Extract input_ids array from params
        if (!params.contains("input_ids")) {
            throw std::runtime_error("Missing input_ids in chat_prefill request");
        }

        std::vector<int32_t> input_ids = params.at("input_ids").as_int_array();
        if (input_ids.empty()) {
            throw std::runtime_error("Empty input_ids");
        }

        int64_t seq_len = input_ids.size();
        std::cout << "[Session] Prefilling " << seq_len << " tokens" << std::endl;

        // Create embedding tensor [seq_len, hidden_size]
        Tensor prompt_hidden({seq_len, lm_cfg_.hidden_size}, DType::Float16);
        prompt_hidden.allocate();

        // Lookup embeddings for each token
        for (int64_t i = 0; i < seq_len; i++) {
            std::vector<int32_t> single_id = {input_ids[i]};
            Tensor single_emb({1, lm_cfg_.hidden_size}, DType::Float16);
            single_emb.allocate();

            embedding_lookup(lm_weights_.embed, single_id, single_emb, ctx_->stream());

            // Copy to prompt_hidden[i, :]
            int64_t offset = i * lm_cfg_.hidden_size * sizeof(uint16_t);
            aclrtMemcpyAsync(
                static_cast<char*>(prompt_hidden.data()) + offset,
                lm_cfg_.hidden_size * sizeof(uint16_t),
                single_emb.data(),
                lm_cfg_.hidden_size * sizeof(uint16_t),
                ACL_MEMCPY_DEVICE_TO_DEVICE,
                ctx_->stream()
            );
        }

        aclrtSynchronizeStream(ctx_->stream());

        // Run prefill
        last_hidden_ = prefill_from_embeddings(
            prompt_hidden,
            lm_weights_,
            lm_cfg_,
            cos_table_,
            sin_table_,
            *decode_state_,
            ctx_->stream()
        );

        aclrtSynchronizeStream(ctx_->stream());
        prefill_done_ = true;

        std::cout << "[Session] Prefill complete, seq_len=" << decode_state_->seq_len << std::endl;

        json::Value result = json::Value::object();
        result["prompt"] = json::Value("Prefill complete");
        result["num_tokens"] = json::Value(static_cast<double>(seq_len));
        return result;
    }

    json::Value chat_generate(const json::Value& params) {
        if (!model_loaded_) {
            throw std::runtime_error("Model not loaded");
        }
        if (!prefill_done_) {
            throw std::runtime_error("Must call chat_prefill before chat_generate");
        }

        // Extract parameters
        int max_new_tokens = 256;
        if (params.contains("max_new_tokens")) {
            max_new_tokens = params.at("max_new_tokens").as_int();
        }

        std::cout << "[Session] Generating up to " << max_new_tokens << " tokens" << std::endl;

        if (minicpmo::profiling_enabled()) {
            minicpmo::profile_reset();
        }

        // Clear streaming state
        stream_chunks_.clear();

        // Get first token from last_hidden using lm_head
        int64_t prev_token = lm_head_greedy(
            last_hidden_,
            lm_weights_,
            lm_cfg_,
            ctx_->stream()
        );

        std::vector<int32_t> generated_ids;
        generated_ids.push_back(static_cast<int32_t>(prev_token));
        const int32_t eos_token_id = 151645;  // <|im_end|> per config.json

        // Check if first token is EOS
        if (prev_token == eos_token_id) {
            std::cout << "[Session] EOS token generated immediately" << std::endl;
        } else {
            // Generate remaining tokens
            for (int i = 1; i < max_new_tokens; i++) {
                int64_t next_token = decode_step_greedy(
                    prev_token,
                    lm_weights_,
                    lm_cfg_,
                    cos_table_,
                    sin_table_,
                    *decode_state_,
                    ctx_->stream()
                );

                generated_ids.push_back(static_cast<int32_t>(next_token));

                // Check for EOS
                if (next_token == eos_token_id) {
                    std::cout << "[Session] EOS token generated at position " << i << std::endl;
                    break;
                }

                prev_token = next_token;
            }
        }

        aclrtSynchronizeStream(ctx_->stream());
        prefill_done_ = false;  // chat_generate is one-shot per turn: call prefill again for the next turn

        if (minicpmo::profiling_enabled()) {
            minicpmo::profile_print();
        }

        json::Value result = json::Value::object();
        result["finished"] = json::Value(true);
        result["num_generated"] = json::Value(static_cast<double>(generated_ids.size()));

        // Return token IDs array for Python-side detokenization
        json::Value ids_array = json::Value::array();
        for (int32_t id : generated_ids) {
            ids_array.arr_val.push_back(json::Value(static_cast<double>(id)));
        }
        result["token_ids"] = ids_array;

        return result;
    }

    json::Value chat_streaming_generate(const json::Value& params) {
        if (!model_loaded_) {
            throw std::runtime_error("Model not loaded");
        }
        if (!prefill_done_) {
            throw std::runtime_error("Must call chat_prefill before chat_streaming_generate");
        }

        // Extract parameters
        int max_new_tokens = 256;
        if (params.contains("max_new_tokens")) {
            max_new_tokens = params.at("max_new_tokens").as_int();
        }

        std::cout << "[Session] Starting streaming generation, max_tokens=" << max_new_tokens << std::endl;

        // Initialize streaming state
        streaming_active_ = true;
        stream_chunks_.clear();

        // Get first token from last_hidden using lm_head
        int64_t prev_token = lm_head_greedy(
            last_hidden_,
            lm_weights_,
            lm_cfg_,
            ctx_->stream()
        );

        // Store first chunk
        const int32_t eos_token_id = 151645;  // <|im_end|> per config.json
        json::Value chunk = json::Value::object();
        chunk["token_id"] = json::Value(static_cast<double>(prev_token));
        chunk["is_eos"] = json::Value(prev_token == eos_token_id);
        stream_chunks_.push_back(chunk);

        // Check if first token is EOS
        if (prev_token != eos_token_id) {
            // Generate remaining tokens
            for (int i = 1; i < max_new_tokens; i++) {
                int64_t next_token = decode_step_greedy(
                    prev_token,
                    lm_weights_,
                    lm_cfg_,
                    cos_table_,
                    sin_table_,
                    *decode_state_,
                    ctx_->stream()
                );

                // Store chunk
                json::Value chunk = json::Value::object();
                chunk["token_id"] = json::Value(static_cast<double>(next_token));
                chunk["is_eos"] = json::Value(next_token == eos_token_id);
                stream_chunks_.push_back(chunk);

                if (next_token == eos_token_id) {
                    std::cout << "[Session] EOS token generated at position " << i << std::endl;
                    break;
                }

                prev_token = next_token;
            }
        }

        aclrtSynchronizeStream(ctx_->stream());
        streaming_active_ = false;
        prefill_done_ = false;  // Reset for next turn

        json::Value result = json::Value::object();
        result["status"] = json::Value("streaming_ready");
        result["num_chunks"] = json::Value(static_cast<double>(stream_chunks_.size()));
        return result;
    }

    json::Value get_next_chunk() {
        json::Value result = json::Value::object();

        if (!stream_chunks_.empty()) {
            result["chunk"] = stream_chunks_.front();
            stream_chunks_.erase(stream_chunks_.begin());
            result["done"] = json::Value(false);
        } else {
            result["chunk"] = json::Value::object();
            result["done"] = json::Value(true);
            streaming_active_ = false;
        }

        return result;
    }

    json::Value metrics() {
        json::Value result = json::Value::object();
        result["backend"] = json::Value("orangepi");
        result["device_id"] = json::Value(static_cast<double>(device_id_));
        result["model_loaded"] = json::Value(model_loaded_);
        return result;
    }

private:
    int device_id_;
    std::unique_ptr<AclContext> ctx_;
    std::unique_ptr<WeightsIndex> weights_index_;

    LanguageModelConfig lm_cfg_;
    VisionConfig vision_cfg_;
    LanguageModelWeights lm_weights_;
    VisionWeights vision_weights_;

    Tensor cos_table_;
    Tensor sin_table_;
    std::unique_ptr<DecodeState> decode_state_;

    // Generation state
    Tensor last_hidden_;  // [1, hidden_size] from prefill
    bool prefill_done_{false};

    bool model_loaded_{false};
    bool streaming_active_{false};
    std::vector<json::Value> stream_chunks_;
};

class BackendServer {
public:
    BackendServer(const std::string& host, int port, int device_id)
        : host_(host), port_(port), device_id_(device_id) {}

    void start() {
        // Create socket
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        // Set socket options
        int opt = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        // Bind
        struct sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr(host_.c_str());
        addr.sin_port = htons(port_);

        if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(server_fd_);
            throw std::runtime_error("Failed to bind to " + host_ + ":" + std::to_string(port_));
        }

        // Listen
        if (listen(server_fd_, 5) < 0) {
            close(server_fd_);
            throw std::runtime_error("Failed to listen");
        }

        std::cout << "[Server] Listening on " << host_ << ":" << port_ << std::endl;

        // Accept connections
        while (running_) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);

            if (client_fd < 0) {
                if (running_) {
                    std::cerr << "[Server] Accept failed" << std::endl;
                }
                continue;
            }

            std::cout << "[Server] Client connected" << std::endl;
            handle_client(client_fd);
        }

        close(server_fd_);
    }

    void stop() {
        running_ = false;
        if (server_fd_ >= 0) {
            shutdown(server_fd_, SHUT_RDWR);
        }
    }

private:
    void handle_client(int client_fd) {
        BackendSession session(device_id_);

        while (running_) {
            // Read 4-byte length prefix (little-endian)
            uint32_t msg_len = 0;
            ssize_t n = recv(client_fd, &msg_len, 4, MSG_WAITALL);
            if (n != 4) {
                break;  // Connection closed or error
            }

            // Allocate buffer for message
            std::vector<char> buffer(msg_len + 1);
            n = recv(client_fd, buffer.data(), msg_len, MSG_WAITALL);
            if (n != static_cast<ssize_t>(msg_len)) {
                break;  // Connection closed or error
            }
            buffer[msg_len] = '\0';

            // Process request
            std::string request(buffer.data(), msg_len);
            std::string response = process_request(session, request);

            // Send response with length prefix
            uint32_t response_len = response.size();
            std::cout << "[Server] Sending response (" << response_len << " bytes)..." << std::endl;
            ssize_t sent1 = send(client_fd, &response_len, 4, MSG_NOSIGNAL);
            if (sent1 < 0) {
                std::cerr << "[Server] Failed to send response header: " << strerror(errno) << std::endl;
                break;
            }
            ssize_t sent2 = send(client_fd, response.c_str(), response_len, MSG_NOSIGNAL);
            if (sent2 < 0) {
                std::cerr << "[Server] Failed to send response body: " << strerror(errno) << std::endl;
                break;
            }
            std::cout << "[Server] Response sent (header: " << sent1 << " bytes, body: " << sent2 << " bytes)" << std::endl;
        }

        close(client_fd);
        std::cout << "[Server] Client disconnected" << std::endl;
    }

    std::string process_request(BackendSession& session, const std::string& request) {
        try {
            std::cout << "[Server] Received request: " << request.substr(0, 100) << "..." << std::endl;
            json::Value req = json::loads(request);
            std::cout << "[Server] JSON parsed successfully" << std::endl;
            std::string type = req["type"].as_string();

            std::cout << "[Server] Request type: " << type << std::endl;

            json::Value response = json::Value::object();

            if (type == "init") {
                std::string model_path = req["model_path"].as_string();
                std::cout << "[Server] Loading model from: " << model_path << std::endl;
                session.load_model(model_path);
                std::cout << "[Server] Model loaded, preparing response" << std::endl;
                response["status"] = json::Value("ok");
                std::cout << "[Server] Response prepared" << std::endl;
            } else if (type == "chat_prefill") {
                json::Value result = session.chat_prefill(req);
                response["status"] = json::Value("ok");
                response["prompt"] = result["prompt"];
            } else if (type == "chat_generate") {
                json::Value result = session.chat_generate(req);
                response["status"] = json::Value("ok");
                response["result"] = result;
            } else if (type == "chat_streaming_generate") {
                json::Value result = session.chat_streaming_generate(req);
                response["status"] = json::Value("ok");
            } else if (type == "get_next_chunk") {
                json::Value chunk = session.get_next_chunk();
                response["status"] = json::Value("ok");
                response["chunk"] = chunk["chunk"];
                response["done"] = chunk["done"];
            } else if (type == "metrics") {
                json::Value result = session.metrics();
                response["status"] = json::Value("ok");
                response["metrics"] = result;
            } else if (type == "shutdown") {
                response["status"] = json::Value("ok");
                running_ = false;
            } else {
                response["status"] = json::Value("error");
                response["error"] = json::Value("Unknown request type: " + type);
            }

            std::cout << "[Server] Serializing response..." << std::endl;
            std::string response_str = json::dumps(response);
            std::cout << "[Server] Response: " << response_str << std::endl;
            return response_str;

        } catch (const std::exception& e) {
            std::cerr << "[Server] Exception in process_request: " << e.what() << std::endl;
            json::Value response = json::Value::object();
            response["status"] = json::Value("error");
            response["error"] = json::Value(std::string(e.what()));
            return json::dumps(response);
        }
    }

    std::string host_;
    int port_;
    int device_id_;
    int server_fd_{-1};
    bool running_{true};
};

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 50051;
    int device_id = 0;

    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--device_id" && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        }
    }

    try {
        std::cout << "MiniCPM-O Backend Server" << std::endl;
        std::cout << "Device: NPU " << device_id << std::endl;

        BackendServer server(host, port, device_id);
        server.start();

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
