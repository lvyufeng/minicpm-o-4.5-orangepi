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
    enum Type { Object, String, Number, Boolean, Null } type;
    std::string str_val;
    double num_val;
    bool bool_val;
    std::map<std::string, Value> obj_val;

    Value() : type(Null) {}
    explicit Value(const std::string& s) : type(String), str_val(s) {}
    explicit Value(const char* s) : type(String), str_val(s) {}
    explicit Value(double n) : type(Number), num_val(n) {}
    explicit Value(bool b) : type(Boolean), bool_val(b) {}

    static Value object() { Value v; v.type = Object; return v; }

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

    // Extract common fields
    extract_string("type");
    extract_string("model_path");
    extract_string("session_id");
    extract_string("method");

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
        model_loaded_ = true;
    }

    json::Value chat_prefill(const json::Value& params) {
        if (!model_loaded_) {
            throw std::runtime_error("Model not loaded");
        }

        // TODO: Parse input (text, images, audio) from params
        // For now just return dummy response
        json::Value result = json::Value::object();
        result["prompt"] = json::Value("User prompt processed");
        return result;
    }

    json::Value chat_generate(const json::Value& params) {
        if (!model_loaded_) {
            throw std::runtime_error("Model not loaded");
        }

        // TODO: Implement actual generation
        // For now just return dummy response
        json::Value result = json::Value::object();
        result["text"] = json::Value("Hello from MiniCPM-O!");
        result["finished"] = json::Value(true);
        return result;
    }

    json::Value chat_streaming_generate(const json::Value& params) {
        if (!model_loaded_) {
            throw std::runtime_error("Model not loaded");
        }

        // Initialize streaming state
        streaming_active_ = true;
        stream_chunks_.clear();

        // TODO: Start actual streaming generation
        // For now, prepare dummy chunks
        json::Value chunk1 = json::Value::object();
        chunk1["text"] = json::Value("Hello ");
        stream_chunks_.push_back(chunk1);

        json::Value chunk2 = json::Value::object();
        chunk2["text"] = json::Value("from MiniCPM-O!");
        stream_chunks_.push_back(chunk2);

        return json::Value::object();
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
            send(client_fd, &response_len, 4, 0);
            send(client_fd, response.c_str(), response_len, 0);
        }

        close(client_fd);
        std::cout << "[Server] Client disconnected" << std::endl;
    }

    std::string process_request(BackendSession& session, const std::string& request) {
        try {
            json::Value req = json::loads(request);
            std::string type = req["type"].as_string();

            std::cout << "[Server] Request: " << type << std::endl;

            json::Value response = json::Value::object();

            if (type == "init") {
                std::string model_path = req["model_path"].as_string();
                session.load_model(model_path);
                response["status"] = json::Value("ok");
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

            return json::dumps(response);

        } catch (const std::exception& e) {
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
