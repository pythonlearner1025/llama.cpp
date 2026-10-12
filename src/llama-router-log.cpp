#include "llama-router-log.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-version.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

static uint64_t router_log_max_bytes(double max_gb) {
    if (!std::isfinite(max_gb) || max_gb <= 0.0 || max_gb > std::numeric_limits<uint64_t>::max() / 1e9) {
        throw std::invalid_argument("router-log-max-gb must be a finite positive value");
    }
    return (uint64_t) (max_gb * 1e9);
}

static std::string json_quote(const std::string & s) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int) c << std::dec;
                } else {
                    out << (char) c;
                }
        }
    }
    out << '"';
    return out.str();
}

static std::vector<std::string> process_argv() {
    std::vector<std::string> result;
#ifdef __linux__
    std::ifstream in("/proc/self/cmdline", std::ios::binary);
    std::string arg;
    while (std::getline(in, arg, '\0')) {
        result.push_back(arg);
    }
#endif
    return result;
}

static std::string model_path_from_argv(const std::vector<std::string> & argv) {
    for (size_t i = 0; i + 1 < argv.size(); ++i) {
        if (argv[i] == "-m" || argv[i] == "--model") {
            return argv[i + 1];
        }
        if (argv[i].rfind("--model=", 0) == 0) {
            return argv[i].substr(8);
        }
    }
    return "unknown";
}

static std::string iso8601_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = {};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

template<typename T>
static void write_npy_atomic(const fs::path & path, const char * descr,
                             const std::vector<size_t> & shape, const std::vector<T> & data) {
    std::ostringstream shape_s;
    shape_s << '(';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) shape_s << ", ";
        shape_s << shape[i];
    }
    if (shape.size() == 1) shape_s << ',';
    shape_s << ')';

    std::string header = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': " + shape_s.str() + ", }";
    const size_t prefix = 10; // magic + version + uint16 header length
    const size_t padded = ((prefix + header.size() + 1 + 63) / 64) * 64 - prefix;
    header.append(padded - header.size() - 1, ' ');
    header.push_back('\n');
    if (header.size() > 65535) throw std::runtime_error("router-log npy header too large");

    const fs::path tmp = path.string() + ".tmp";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    const uint8_t version[2] = {1, 0};
    out.write((const char *) version, 2);
    const uint16_t len = (uint16_t) header.size();
    const uint8_t len_le[2] = {(uint8_t) len, (uint8_t) (len >> 8)};
    out.write((const char *) len_le, 2);
    out.write(header.data(), header.size());
    out.write((const char *) data.data(), data.size() * sizeof(T));
    out.close();
    if (!out) throw std::runtime_error("failed writing " + tmp.string());
    fs::rename(tmp, path);
}

struct llama_router_log::impl {
    static constexpr size_t chunk_tokens = 4096;

    fs::path dir;
    std::string arch;
    const size_t n_layers;
    const size_t n_experts;
    const size_t k;
    const uint64_t max_bytes;
    uint64_t bytes_per_token = 0;
    std::vector<std::string> argv;
    std::string created;
    std::string model_path;
    std::vector<size_t> chunk_sizes;
    std::vector<ggml_fp16_t> logits;
    std::vector<int32_t> token_id, pos, seq_id, ubatch_id, ubatch_n_tokens;
    std::vector<uint8_t> is_decode;
    uint64_t bytes_written = 0;
    int32_t next_ubatch_id = 0;
    bool enabled = true;
    bool warned = false;
    std::mutex mutex;

    struct pending_ubatch {
        size_t n_tokens = 0;
        size_t layer_bytes = 0;
        ggml_type type = GGML_TYPE_COUNT;
        int32_t id = 0;
        bool decode = false;
        // float storage guarantees alignment for both supported source types;
        // the allocation is rounded up to cover the byte extent.
        std::vector<float> raw;
        std::vector<int32_t> token_id, pos, seq_id;
    };
    std::vector<pending_ubatch> pending;
    size_t pending_count = 0;
    size_t pending_tokens = 0;
    std::vector<ggml_tensor *> tensors;

    impl(const std::string & path, double max_gb, const llama_model & m) :
        dir(path), arch(m.arch_name()), n_layers(m.hparams.n_layer()), n_experts(m.hparams.n_expert),
        k(m.hparams.n_expert_used()), max_bytes(router_log_max_bytes(max_gb)),
        argv(process_argv()), created(iso8601_now()),
        model_path(m.path_model.empty() ? model_path_from_argv(argv) : m.path_model) {
        if (n_layers == 0 || n_experts == 0) {
            throw std::runtime_error("router logging requires an MoE model");
        }
        if (n_layers > std::numeric_limits<size_t>::max() / n_experts ||
                n_layers * n_experts > (std::numeric_limits<size_t>::max() - 21) / (2 * chunk_tokens)) {
            throw std::runtime_error("router-log model dimensions are too large");
        }
        bytes_per_token = 2 * n_layers * n_experts + 21;
        fs::create_directories(dir);
        if (fs::exists(dir / "manifest.json")) {
            throw std::runtime_error("router-log directory already contains manifest.json: " + dir.string());
        }
        logits.reserve(std::min<size_t>(chunk_tokens, 128) * n_layers * n_experts);
        token_id.reserve(chunk_tokens); pos.reserve(chunk_tokens); seq_id.reserve(chunk_tokens);
        ubatch_id.reserve(chunk_tokens); ubatch_n_tokens.reserve(chunk_tokens); is_decode.reserve(chunk_tokens);
        tensors.resize(n_layers);
        write_manifest();
        LLAMA_LOG_INFO("router-log: enabled at %s (cap %.3f GB)\n", dir.string().c_str(), max_gb);
    }

    ~impl() {
        try {
            flush();
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("router-log: final flush failed: %s\n", e.what());
        }
    }

    std::string prefix(size_t i) const {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "chunk-%06zu", i);
        return buf;
    }

    void write_manifest() {
        const fs::path tmp = dir / "manifest.json.tmp";
        std::ofstream out(tmp, std::ios::trunc);
        const bool qwen4exp = arch == "qwen4exp";
        out << "{\n  \"version\": 2,\n"
            << "  \"model\": " << json_quote(model_path) << ",\n"
            << "  \"arch\": " << json_quote(arch) << ",\n"
            << "  \"n_layers\": " << n_layers << ",\n"
            << "  \"n_experts\": " << n_experts << ",\n"
            << "  \"k\": " << k << ",\n"
            << "  \"gating\": {\"function\": " << json_quote(qwen4exp ? "softmax" : "unknown")
            << ", \"router_bias\": " << (qwen4exp ? "false" : "null")
            << ", \"selection_bias\": " << (qwen4exp ? "false" : "null")
            << ", \"top_k_normalized\": " << (qwen4exp ? "true" : "null") << "},\n"
            << "  \"layer_index_map\": [";
        for (size_t i = 0; i < n_layers; ++i) out << (i ? ", " : "") << i;
        out << "],\n  \"llama_cpp_commit\": " << json_quote(LLAMA_COMMIT) << ",\n"
            << "  \"dtype\": \"float16\",\n  \"chunk_tokens\": " << chunk_tokens << ",\n"
            << "  \"created\": " << json_quote(created) << ",\n  \"command\": [";
        for (size_t i = 0; i < argv.size(); ++i) out << (i ? ", " : "") << json_quote(argv[i]);
        out << "],\n  \"chunks\": [";
        for (size_t i = 0; i < chunk_sizes.size(); ++i) {
            out << (i ? ", " : "") << "{\"prefix\": " << json_quote(prefix(i))
                << ", \"n_tokens\": " << chunk_sizes[i] << "}";
        }
        out << "]\n}\n";
        out.close();
        if (!out) throw std::runtime_error("failed writing router-log manifest");
        fs::rename(tmp, dir / "manifest.json");
    }

    void stop_at_cap() {
        enabled = false;
        if (!warned) {
            warned = true;
            LLAMA_LOG_WARN("router-log: size cap reached after %.3f GB; logging stopped\n", bytes_written / 1e9);
        }
    }

    bool exceeds_cap(size_t n_tokens) const {
        constexpr uint64_t file_overhead = 8192;
        if (bytes_written > max_bytes || file_overhead > max_bytes - bytes_written) {
            return true;
        }
        return n_tokens > (max_bytes - bytes_written - file_overhead) / bytes_per_token;
    }

    void flush() {
        if (token_id.empty()) return;
        const size_t i = chunk_sizes.size();
        const size_t n = token_id.size();
        const std::string pfx = prefix(i);
        write_npy_atomic(dir / (pfx + ".logits.npy"), "<f2", {n, n_layers, n_experts}, logits);
        write_npy_atomic(dir / (pfx + ".token_id.npy"), "<i4", {n}, token_id);
        write_npy_atomic(dir / (pfx + ".pos.npy"), "<i4", {n}, pos);
        write_npy_atomic(dir / (pfx + ".seq_id.npy"), "<i4", {n}, seq_id);
        write_npy_atomic(dir / (pfx + ".ubatch_id.npy"), "<i4", {n}, ubatch_id);
        write_npy_atomic(dir / (pfx + ".ubatch_n_tokens.npy"), "<i4", {n}, ubatch_n_tokens);
        write_npy_atomic(dir / (pfx + ".is_decode.npy"), "|b1", {n}, is_decode);
        for (const char * suffix : {".logits.npy", ".token_id.npy", ".pos.npy", ".seq_id.npy",
                                   ".ubatch_id.npy", ".ubatch_n_tokens.npy", ".is_decode.npy"}) {
            bytes_written += fs::file_size(dir / (pfx + suffix));
        }
        chunk_sizes.push_back(n);
        logits.clear(); token_id.clear(); pos.clear(); seq_id.clear();
        ubatch_id.clear(); ubatch_n_tokens.clear(); is_decode.clear();
        write_manifest();
    }

    bool capture(llama_context & ctx, const llm_graph_result & graph, const llama_ubatch & ubatch) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!enabled) return false;
        if (pending_tokens > std::numeric_limits<size_t>::max() - token_id.size() ||
                ubatch.n_tokens > std::numeric_limits<size_t>::max() - token_id.size() - pending_tokens) {
            stop_at_cap();
            return false;
        }
        if (exceeds_cap(token_id.size() + pending_tokens + ubatch.n_tokens)) {
            // Pending copies cannot be consumed until the caller's normal
            // scheduler synchronization.
            if (pending_tokens > 0) {
                stop_at_cap();
                return false;
            }
            flush();
            if (exceeds_cap(ubatch.n_tokens)) {
                stop_at_cap();
                return false;
            }
        }

        std::fill(tensors.begin(), tensors.end(), nullptr);
        size_t layer_bytes = 0;
        ggml_type tensor_type = GGML_TYPE_COUNT;
        for (size_t il = 0; il < n_layers; ++il) {
            const std::string name = "ffn_moe_logits_log-" + std::to_string(il);
            tensors[il] = ggml_graph_get_tensor(graph.get_gf(), name.c_str());
            if (!tensors[il]) {
                LLAMA_LOG_ERROR("router-log: missing graph tensor %s; logging stopped\n", name.c_str());
                enabled = false;
                return false;
            }
            if ((size_t) tensors[il]->ne[0] != n_experts || (size_t) tensors[il]->ne[1] != ubatch.n_tokens ||
                    (size_t) ggml_nelements(tensors[il]) != n_experts * ubatch.n_tokens) {
                LLAMA_LOG_ERROR("router-log: unexpected shape for %s; logging stopped\n", name.c_str());
                enabled = false;
                return false;
            }
            if (il == 0) {
                tensor_type = tensors[il]->type;
                layer_bytes = ggml_nbytes(tensors[il]);
            }
            if (!ggml_is_contiguous(tensors[il]) ||
                    tensors[il]->type != tensor_type || ggml_nbytes(tensors[il]) != layer_bytes ||
                    (tensor_type != GGML_TYPE_F32 && tensor_type != GGML_TYPE_F16)) {
                LLAMA_LOG_ERROR("router-log: unsupported/inconsistent logits dtype; logging stopped\n");
                enabled = false;
                return false;
            }
        }

        if (pending_count == pending.size()) {
            pending.emplace_back();
        }
        auto & item = pending[pending_count];
        item.n_tokens = ubatch.n_tokens;
        item.layer_bytes = layer_bytes;
        item.type = tensor_type;
        item.id = next_ubatch_id++;
        item.decode = ubatch.n_seq_tokens == 1;
        if (layer_bytes > (std::numeric_limits<size_t>::max() - sizeof(float) + 1) / n_layers) {
            throw std::runtime_error("router-log ubatch is too large");
        }
        item.raw.resize((layer_bytes * n_layers + sizeof(float) - 1) / sizeof(float));
        item.token_id.resize(ubatch.n_tokens);
        item.pos.resize(ubatch.n_tokens);
        item.seq_id.resize(ubatch.n_tokens);
        for (size_t t = 0; t < ubatch.n_tokens; ++t) {
            item.token_id[t] = ubatch.token ? ubatch.token[t] : LLAMA_TOKEN_NULL;
            item.pos[t] = ubatch.pos ? ubatch.pos[t] : -1;
            item.seq_id[t] = ubatch.n_seq_id[t] > 0 ? ubatch.seq_id[t][0] : -1;
        }
        for (size_t il = 0; il < n_layers; ++il) {
            ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(ctx.get_sched(), tensors[il]);
            auto * dst = reinterpret_cast<uint8_t *>(item.raw.data()) + il * layer_bytes;
            if (ggml_backend_buffer_is_host(tensors[il]->buffer)) {
                // CPU graph execution is synchronous. Calling the generic
                // async getter here falls back to a backend synchronize for
                // every layer, which fragmented the intended single sync.
                std::memcpy(dst, tensors[il]->data, layer_bytes);
            } else {
                ggml_backend_tensor_get_async(backend, tensors[il], dst, 0, layer_bytes);
            }
        }
        ++pending_count;
        pending_tokens += ubatch.n_tokens;
        return true;
    }

    void drain() {
        std::lock_guard<std::mutex> lock(mutex);
        for (size_t ip = 0; ip < pending_count; ++ip) {
            auto & item = pending[ip];
            const size_t nt = item.n_tokens;
            size_t src_token = 0;
            while (src_token < nt) {
                const size_t take = std::min(nt - src_token, chunk_tokens - token_id.size());
                const size_t old = token_id.size();
                logits.resize((old + take) * n_layers * n_experts);
                for (size_t q = 0; q < take; ++q) {
                    const size_t t = src_token + q;
                    for (size_t il = 0; il < n_layers; ++il) {
                        const size_t src_base = il * nt * n_experts + t * n_experts;
                        const size_t dst_base = ((old + q) * n_layers + il) * n_experts;
                        if (item.type == GGML_TYPE_F32) {
                            const float * src = item.raw.data();
                            ggml_fp32_to_fp16_row(src + src_base, logits.data() + dst_base, n_experts);
                        } else {
                            const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(item.raw.data());
                            std::memcpy(logits.data() + dst_base, src + src_base, n_experts * sizeof(ggml_fp16_t));
                        }
                    }
                    token_id.push_back(item.token_id[t]);
                    pos.push_back(item.pos[t]);
                    seq_id.push_back(item.seq_id[t]);
                    ubatch_id.push_back(item.id);
                    ubatch_n_tokens.push_back((int32_t) nt);
                    is_decode.push_back(item.decode);
                }
                src_token += take;
                if (token_id.size() == chunk_tokens) {
                    flush();
                }
            }
            const size_t retained_limit = n_layers * n_experts * 32 * sizeof(float);
            if (item.raw.capacity() * sizeof(float) > retained_limit) {
                std::vector<float>().swap(item.raw);
            }
        }
        pending_count = 0;
        pending_tokens = 0;
    }

    void log_request(int32_t sid, int64_t task, const char * request, int32_t prompt_tokens,
                     int64_t start_ms, int64_t end_ms) {
        std::lock_guard<std::mutex> lock(mutex);
        std::ofstream out(dir / "requests.jsonl", std::ios::app);
        out << "{\"seq_id\":" << sid << ",\"slot_id\":" << sid << ",\"task_id\":" << task
            << ",\"request_id\":" << json_quote(request ? request : "")
            << ",\"prompt_token_count\":" << prompt_tokens
            << ",\"start_unix_ms\":" << start_ms << ",\"end_unix_ms\":" << end_ms << "}\n";
        if (!out) {
            throw std::runtime_error("failed writing router-log request sidecar");
        }
    }

    void disable_after_error() {
        std::lock_guard<std::mutex> lock(mutex);
        enabled = false;
        pending.clear(); pending_count = 0; pending_tokens = 0;
        logits.clear(); token_id.clear(); pos.clear(); seq_id.clear();
        ubatch_id.clear(); ubatch_n_tokens.clear(); is_decode.clear();
    }

    void disable() {
        std::lock_guard<std::mutex> lock(mutex);
        enabled = false;
    }
};

llama_router_log::llama_router_log(const std::string & dir, double max_gb, const llama_model & model) :
    p(std::make_unique<impl>(dir, max_gb, model)) {}
llama_router_log::~llama_router_log() = default;

bool llama_router_log::capture(llama_context & ctx, const llm_graph_result & graph, const llama_ubatch & ubatch) {
    try {
        return p->capture(ctx, graph, ubatch);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("router-log: capture failed; logging stopped: %s\n", e.what());
        // Keep pending buffers alive until the next scheduler synchronization.
        p->disable();
        return false;
    }
}

void llama_router_log::drain() {
    try {
        p->drain();
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("router-log: copy completion failed; logging stopped: %s\n", e.what());
        p->disable_after_error();
    }
}

void llama_router_log::flush() {
    std::lock_guard<std::mutex> lock(p->mutex);
    try {
        p->flush();
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("router-log: flush failed; logging stopped: %s\n", e.what());
        p->enabled = false;
        p->logits.clear(); p->token_id.clear(); p->pos.clear(); p->seq_id.clear();
        p->ubatch_id.clear(); p->ubatch_n_tokens.clear(); p->is_decode.clear();
    }
}

void llama_router_log::log_request(int32_t seq_id, int64_t task_id, const char * request_id,
                                   int32_t prompt_tokens, int64_t start_unix_ms, int64_t end_unix_ms) {
    try {
        p->log_request(seq_id, task_id, request_id, prompt_tokens, start_unix_ms, end_unix_ms);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("router-log: request sidecar write failed: %s\n", e.what());
    }
}

bool llama_router_log::matches(const llama_model & model) const {
    return p->arch == model.arch_name() && p->n_layers == model.hparams.n_layer() &&
           p->n_experts == model.hparams.n_expert && p->model_path == model.path_model;
}

std::shared_ptr<llama_router_log> llama_router_log_get(
        const std::string & dir, double max_gb, const llama_model & model) {
    static std::mutex registry_mutex;
    static std::map<std::string, std::shared_ptr<llama_router_log>> registry;
    std::lock_guard<std::mutex> lock(registry_mutex);
    const std::string key = fs::absolute(dir).lexically_normal().string();
    if (auto existing = registry[key]) {
        if (!existing->matches(model)) {
            throw std::runtime_error("router-log directory requested by incompatible models: " + key);
        }
        return existing;
    }
    auto result = std::make_shared<llama_router_log>(key, max_gb, model);
    registry[key] = result;
    return result;
}
