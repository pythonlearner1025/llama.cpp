#include "arg.h"
#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "ggml.h"
#include "json.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct route_capture {
    int n_layers = 0;
    int k = 0;
    int turn = 0;
    int session = 0;
    bool decode = false;
    std::vector<uint16_t> experts;
    std::vector<ggml_fp16_t> gates;
    std::vector<std::vector<uint16_t>> expert_layers;
    std::vector<std::vector<ggml_fp16_t>> gate_layers;
    std::vector<uint8_t> is_decode;
    std::vector<int32_t> turn_id;
    std::vector<int32_t> session_id;

    void begin(size_t n, bool dec, int tid) {
        decode = dec; turn = tid;
        is_decode.insert(is_decode.end(), n, dec);
        turn_id.insert(turn_id.end(), n, tid);
        session_id.insert(session_id.end(), n, session);
    }
};

static bool route_cb(ggml_tensor * t, bool ask, void * data) {
    const bool route_prefix = std::strncmp(t->name, "ffn_moe_topk-", 13) == 0 ||
        std::strncmp(t->name, "ffn_moe_weights_norm-", 21) == 0;
    int il = -1, used = 0;
    bool topk = std::sscanf(t->name, "ffn_moe_topk-%d%n", &il, &used) == 1 && t->name[used] == '\0';
    used = 0;
    bool weights = std::sscanf(t->name, "ffn_moe_weights_norm-%d%n", &il, &used) == 1 && t->name[used] == '\0';
    if (ask) return route_prefix;
    if (!topk && !weights) return true;
    auto & c = *(route_capture *) data;
    if (il < 0 || il >= c.n_layers || ggml_nelements(t) % c.k != 0) {
        std::fprintf(stderr, "unexpected routing tensor %s shape [%lld,%lld]\n", t->name,
                     (long long) t->ne[0], (long long) t->ne[1]);
        std::abort();
    }
    std::vector<uint8_t> raw(ggml_nbytes(t));
    if (ggml_backend_buffer_is_host(t->buffer)) std::memcpy(raw.data(), t->data, raw.size());
    else ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    const size_t rows = ggml_nelements(t)/c.k;
    // A tensor can appear in more than one scheduler split (notably the final
    // layer).  Keep only the rows belonging to tokens submitted so far.
    const size_t have = topk ? c.expert_layers[il].size()/c.k : c.gate_layers[il].size()/c.k;
    if (have >= c.is_decode.size()) return true;
    if (have + rows > c.is_decode.size()) std::abort();
    for (size_t r = 0; r < rows; ++r) for (int j = 0; j < c.k; ++j) {
        const size_t src = r*c.k + j;
        if (topk) {
            if (t->type != GGML_TYPE_I32) std::abort();
            c.expert_layers[il].push_back((uint16_t) ((const int32_t *) raw.data())[src]);
        } else {
            float value;
            if (t->type == GGML_TYPE_F32) value = ((const float *) raw.data())[src];
            else if (t->type == GGML_TYPE_F16) value = ggml_fp16_to_fp32(((const ggml_fp16_t *) raw.data())[src]);
            else std::abort();
            c.gate_layers[il].push_back(ggml_fp32_to_fp16(value));
        }
    }
    return true;
}

static std::string cache_log;
static void log_cb(enum ggml_log_level level, const char * text, void *) {
    common_log_default_callback(level, text, nullptr);
    if (std::strstr(text, "llama_moe_cache:")) cache_log += text;
}

template<typename T>
static void write_npy(const fs::path & path, const char * descr, const std::vector<size_t> & shape,
                      const std::vector<T> & data) {
    std::ostringstream hs;
    hs << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) hs << shape[i] << (i + 1 == shape.size() ? (shape.size() == 1 ? "," : "") : ", ");
    hs << "), }";
    std::string h = hs.str();
    size_t pad = 16 - ((10 + h.size() + 1) % 16); h.append(pad, ' '); h.push_back('\n');
    std::ofstream f(path, std::ios::binary);
    const char magic[] = "\x93NUMPY"; f.write(magic, 6); f.put(1); f.put(0);
    uint16_t len = (uint16_t) h.size(); f.put(len & 255); f.put(len >> 8); f.write(h.data(), h.size());
    f.write((const char *) data.data(), data.size()*sizeof(T));
    if (!f) throw std::runtime_error("failed writing " + path.string());
}

static std::string slurp(const fs::path & p) {
    std::ifstream f(p, std::ios::binary); if (!f) throw std::runtime_error("cannot read " + p.string());
    return std::string(std::istreambuf_iterator<char>(f), {});
}

struct script_turn { std::string role, content; int max_tokens = 300; };

static std::vector<script_turn> read_script(const fs::path & path) {
    std::ifstream f(path); if (!f) throw std::runtime_error("cannot read " + path.string());
    std::vector<script_turn> out; std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        common_json j = common_json::parse(line);
        script_turn t; t.role = j.at("role").get<std::string>();
        if (j.contains("content")) t.content = j.at("content").get<std::string>();
        else if (j.contains("content_file")) t.content = slurp(path.parent_path()/j.at("content_file").get<std::string>());
        if (j.contains("max_tokens")) t.max_tokens = j.at("max_tokens").get<int>();
        out.push_back(std::move(t));
    }
    return out;
}

int main(int argc, char ** argv) try {
    std::string script_path, output_path, prefill_base, prefill_append, prefill_cache, teacher_forced, teacher_script, teacher_raw;
    std::vector<std::string> prefill_appends, prefill_outputs;
    int session_id = 0; bool capture_enabled = true;
    std::vector<char *> pass; pass.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--benchmark-no-capture") {
            capture_enabled = false;
        } else if ((a == "--session-script" || a == "--output" || a == "--session-id" ||
                    a == "--prefill-base" || a == "--prefill-append" || a == "--prefill-cache" ||
                    a == "--teacher-forced" || a == "--teacher-script" || a == "--teacher-raw") && i + 1 < argc) {
            std::string v = argv[++i];
            if (a == "--session-script") script_path = v;
            else if (a == "--output") { output_path = v; prefill_outputs.push_back(v); }
            else if (a == "--prefill-base") prefill_base = v;
            else if (a == "--prefill-append") { prefill_append = v; prefill_appends.push_back(v); }
            else if (a == "--prefill-cache") prefill_cache = v;
            else if (a == "--teacher-forced") teacher_forced = v;
            else if (a == "--teacher-script") teacher_script = v;
            else if (a == "--teacher-raw") teacher_raw = v;
            else session_id = std::stoi(v);
        } else pass.push_back(argv[i]);
    }
    const bool prefill_only = !prefill_base.empty() || !prefill_append.empty();
    const bool teacher_only = !teacher_forced.empty() || !teacher_script.empty() || !teacher_raw.empty();
    if (output_path.empty() || (prefill_only && (prefill_base.empty() || prefill_appends.empty() || prefill_appends.size() != prefill_outputs.size())) ||
            (teacher_only && teacher_raw.empty() && (teacher_forced.empty() || teacher_script.empty())) ||
            (!prefill_only && !teacher_only && script_path.empty())) {
        throw std::runtime_error("use --session-script, --teacher-forced with --teacher-script, or --prefill-base with --prefill-append");
    }

    common_params params; common_init();
    if (!common_params_parse((int) pass.size(), pass.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;
    if (!teacher_raw.empty()) {
        // Runtime enablement cannot retrofit logger-only graph outputs into an
        // already-created graph object, so the forcing pass starts logged.
        params.router_log_dir = output_path.c_str();
    }
    params.warmup = false;
    params.n_outputs_max = 128; // observable prefill chunks; avoids reserving n_batch logits
    route_capture cap; cap.session = session_id;
    if (prefill_only || teacher_only) capture_enabled = false;
    if (capture_enabled) { params.cb_eval = route_cb; params.cb_eval_user_data = &cap; }
    llama_backend_init(); llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    llama_model * model = init->model(); llama_context * ctx = init->context();
    if (!model || !ctx) throw std::runtime_error("model initialization failed");

    if (teacher_only) {
        cap.n_layers = llama_model_n_layer(model);
        cap.k = 10;
        cap.expert_layers.resize(cap.n_layers);
        cap.gate_layers.resize(cap.n_layers);
        std::vector<llama_token> forced_tokens;
        std::vector<uint8_t> forced_decode;
        common_batch teacher_batch(ctx);
        std::vector<llama_token> evaluated;
        if (!teacher_raw.empty()) {
            std::ifstream raw(teacher_raw, std::ios::binary);
            uint64_t n = 0;
            raw.read((char *) &n, sizeof(n));
            if (!raw || n > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("invalid teacher raw header");
            }
            forced_tokens.resize((size_t) n);
            forced_decode.resize((size_t) n);
            raw.read((char *) forced_tokens.data(), forced_tokens.size() * sizeof(llama_token));
            raw.read((char *) forced_decode.data(), forced_decode.size());
            if (!raw) throw std::runtime_error("truncated teacher raw file");
        } else {
        const std::string transcript = slurp(teacher_forced);
        std::vector<std::string> answers;
        const std::string begin_marker = "<|im_start|>assistant\n";
        const std::string end_marker = "<|im_end|>";
        size_t search = 0;
        while ((search = transcript.find(begin_marker, search)) != std::string::npos) {
            const size_t begin = search + begin_marker.size();
            const size_t end = transcript.find(end_marker, begin);
            if (end == std::string::npos) {
                throw std::runtime_error("unterminated assistant message in teacher transcript");
            }
            answers.push_back(transcript.substr(begin, end - begin));
            search = end + end_marker.size();
        }
        auto eval = [&](const std::vector<llama_token> & tokens, size_t batch_size, bool decode) {
            for (size_t off = 0; off < tokens.size(); off += batch_size) {
                const size_t n = std::min(batch_size, tokens.size() - off);
                teacher_batch.clear();
                for (size_t i = 0; i < n; ++i) {
                    teacher_batch.add(tokens[off + i], evaluated.size() + i, 0, true);
                }
                cap.begin(n, decode, 0);
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, teacher_batch.get())) {
                    throw std::runtime_error("teacher-forced decode failed");
                }
                evaluated.insert(evaluated.end(), tokens.begin() + off, tokens.begin() + off + n);
                forced_tokens.insert(forced_tokens.end(), tokens.begin() + off, tokens.begin() + off + n);
                forced_decode.insert(forced_decode.end(), n, decode);
            }
        };
        auto tmpls = common_chat_templates_init(model, params.chat_template);
        std::vector<common_chat_msg> messages;
        std::string pending;
        size_t answer_index = 0;
        const auto turns = read_script(teacher_script);
        for (size_t ti = 0; ti < turns.size(); ++ti) {
            const auto & st = turns[ti];
            if (st.role != "assistant") {
                common_chat_msg msg;
                msg.role = st.role == "tool" ? "user" : st.role;
                msg.content = st.role == "tool" ? "[tool output]\n" + st.content : st.content;
                const bool add_generation_prompt = ti + 1 < turns.size() && turns[ti + 1].role == "assistant";
                if (!evaluated.empty()) {
                    pending += common_chat_format_single(tmpls.get(), messages, msg, add_generation_prompt, params.use_jinja);
                }
                messages.push_back(std::move(msg));
                continue;
            }
            if (answer_index >= answers.size()) {
                throw std::runtime_error("teacher transcript has too few assistant messages");
            }
            if (evaluated.empty()) {
                common_chat_templates_inputs initial;
                initial.messages = messages; initial.add_generation_prompt = true; initial.use_jinja = params.use_jinja;
                pending = common_chat_templates_apply(tmpls.get(), initial).prompt;
            }
            eval(common_tokenize(ctx, pending, evaluated.empty(), true), 128, false);
            pending.clear();
            std::string answer = answers[answer_index];
            const std::string thinking_prefix = "<think>\n\n</think>\n\n";
            if (answer.rfind(thinking_prefix, 0) == 0) {
                answer.erase(0, thinking_prefix.size());
            }
            const auto answer_tokens = common_tokenize(ctx, answer, false, true);
            if ((int) answer_tokens.size() != st.max_tokens) {
                throw std::runtime_error("assistant message " + std::to_string(answer_index) +
                    " retokenized to " + std::to_string(answer_tokens.size()) +
                    " tokens, expected " + std::to_string(st.max_tokens));
            }
            eval(answer_tokens, 1, true);
            // The chat template injects the model's empty thinking preamble
            // around assistant content, so compare the fully rendered session
            // after all turns instead of comparing this raw answer string.
            common_chat_msg answer_msg; answer_msg.role = "assistant"; answer_msg.content = answer;
            ++answer_index;
            messages.push_back(std::move(answer_msg));
        }
        if (answer_index != answers.size()) {
            throw std::runtime_error("teacher transcript has extra assistant messages");
        }
        common_chat_templates_inputs final_in;
        final_in.messages = messages; final_in.add_generation_prompt = false; final_in.use_jinja = params.use_jinja;
        if (common_chat_templates_apply(tmpls.get(), final_in).prompt != transcript) {
            throw std::runtime_error("rendered replay differs from saved transcript");
        }
        const std::string raw_path = output_path + ".teacher.raw";
        std::ofstream raw(raw_path, std::ios::binary | std::ios::trunc);
        const uint64_t n = forced_tokens.size();
        raw.write((const char *) &n, sizeof(n));
        raw.write((const char *) forced_tokens.data(), forced_tokens.size() * sizeof(llama_token));
        raw.write((const char *) forced_decode.data(), forced_decode.size());
        if (!raw) throw std::runtime_error("failed writing " + raw_path);
        std::fprintf(stderr, "verified %zu tokens to %s\n", forced_tokens.size(), raw_path.c_str());
        init.reset();
        llama_backend_free();
        return 0;
        }
        evaluated.clear();
        size_t off = 0;
        while (off < forced_tokens.size()) {
            const bool decode = forced_decode[off] != 0;
            size_t n = 1;
            if (!decode) {
                while (off + n < forced_tokens.size() && !forced_decode[off + n] && n < 128) {
                    ++n;
                }
            }
            teacher_batch.clear();
            for (size_t i = 0; i < n; ++i) {
                teacher_batch.add(forced_tokens[off + i], off + i, 0, true);
            }
            cap.begin(n, decode, 0);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, teacher_batch.get())) {
                throw std::runtime_error("teacher-forced replay failed");
            }
            off += n;
        }
        std::fprintf(stderr, "teacher-forced %zu verified tokens to %s\n", forced_tokens.size(), output_path.c_str());
        init.reset();
        llama_backend_free();
        return 0;
    }

    if (prefill_only) {
        const std::string base = slurp(prefill_base);
        const auto base_tokens = common_tokenize(ctx, base, true, true);
        common_batch prefill_batch(ctx);
        for (size_t job = 0; job < prefill_appends.size(); ++job) {
            const std::string combined = base + "\n\n" + slurp(prefill_appends[job]);
            const auto all_tokens = common_tokenize(ctx, combined, true, true);
            size_t prefix = 0;
            auto eval_range = [&](size_t first, size_t last) {
                for (size_t off = first; off < last; off += params.n_ubatch) {
                    const size_t n = std::min<size_t>(params.n_ubatch, last - off);
                    prefill_batch.clear();
                    for (size_t i = 0; i < n; ++i) {
                        prefill_batch.add(all_tokens[off + i], off + i, 0, off + i + 1 == last);
                    }
                    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, prefill_batch.get())) {
                        throw std::runtime_error("prefill decode failed");
                    }
                }
            };
            llama_memory_clear(llama_get_memory(ctx), true);
            if (!prefill_cache.empty()) {
                std::vector<llama_token> cached(llama_n_ctx(ctx));
                size_t count = 0;
                if (!llama_state_load_file(ctx, prefill_cache.c_str(), cached.data(), cached.size(), &count)) {
                    throw std::runtime_error("failed to load prefill cache " + prefill_cache);
                }
                cached.resize(count);
                while (prefix < cached.size() && prefix < all_tokens.size() && cached[prefix] == all_tokens[prefix]) {
                    ++prefix;
                }
                if (prefix < cached.size() &&
                        !llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) prefix, -1)) {
                    llama_memory_clear(llama_get_memory(ctx), true);
                    eval_range(0, prefix);
                }
            } else {
                while (prefix < base_tokens.size() && prefix < all_tokens.size() &&
                        base_tokens[prefix] == all_tokens[prefix]) {
                    ++prefix;
                }
                eval_range(0, prefix);
            }
            llama_synchronize(ctx);
            if (!llama_router_log_enable(ctx, prefill_outputs[job].c_str(), params.router_log_max_gb)) {
                throw std::runtime_error("failed to enable router logging after base prefill");
            }
            eval_range(prefix, all_tokens.size());
            llama_router_log_disable(ctx);
            std::fprintf(stderr, "logged %zu appended tokens after %zu base tokens to %s\n",
                         all_tokens.size() - prefix, prefix, prefill_outputs[job].c_str());
        }
        init.reset();
        llama_backend_free();
        return 0;
    }
    cap.n_layers = llama_model_n_layer(model);
    cap.k = 10; // validated against the callback tensors
    cap.expert_layers.resize(cap.n_layers); cap.gate_layers.resize(cap.n_layers);
    const int n_experts = 512;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tmpls = common_chat_templates_init(model, params.chat_template);
    std::vector<common_chat_msg> messages;
    std::vector<llama_token> evaluated;
    common_batch batch(ctx);
    llama_sampler * greedy = llama_sampler_init_greedy();
    auto eval = [&](const std::vector<llama_token> & tok, bool decode, int turn) {
        const size_t eval_batch = decode ? 1 : 128;
        for (size_t off = 0; off < tok.size(); off += eval_batch) {
            size_t n = std::min<size_t>(eval_batch, tok.size() - off);
            batch.clear();
            for (size_t q = 0; q < n; ++q) batch.add(tok[off + q], evaluated.size() + q, 0, true);
            cap.begin(n, decode, turn);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) throw std::runtime_error("decode failed");
            evaluated.insert(evaluated.end(), tok.begin() + off, tok.begin() + off + n);
        }
    };
    const auto turns = read_script(script_path);
    std::string pending;
    size_t generated = 0; double decode_seconds = 0;
    for (size_t ti = 0; ti < turns.size(); ++ti) {
        const auto & st = turns[ti];
        if (st.role != "assistant") {
            common_chat_msg msg;
            // Some templates reject a tool result without a preceding structured
            // tool-call object.  Session traces contain results only, so retain
            // their semantics and bytes under a visibly labelled user message.
            msg.role = st.role == "tool" ? "user" : st.role;
            msg.content = st.role == "tool" ? "[tool output]\n" + st.content : st.content;
            const bool add_generation_prompt = ti + 1 < turns.size() && turns[ti + 1].role == "assistant";
            if (!evaluated.empty()) {
                pending += common_chat_format_single(tmpls.get(), messages, msg, add_generation_prompt, params.use_jinja);
            }
            messages.push_back(std::move(msg));
            continue;
        }
        if (evaluated.empty()) {
            common_chat_templates_inputs initial;
            initial.messages = messages; initial.add_generation_prompt = true; initial.use_jinja = params.use_jinja;
            pending = common_chat_templates_apply(tmpls.get(), initial).prompt;
        }
        auto prompt = common_tokenize(ctx, pending, evaluated.empty(), true);
        eval(prompt, false, ti); pending.clear();
        std::string answer;
        auto start = std::chrono::steady_clock::now();
        for (int q = 0; q < st.max_tokens; ++q) {
            llama_token id = llama_sampler_sample(greedy, ctx, -1);
            eval(std::vector<llama_token>{id}, true, ti); ++generated;
            if (llama_vocab_is_eog(vocab, id)) break;
            answer += common_token_to_piece(ctx, id, true);
        }
        decode_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        common_chat_msg answer_msg; answer_msg.role = "assistant"; answer_msg.content = answer;
        messages.push_back(std::move(answer_msg));
        std::fprintf(stderr, "turn %zu generated %zu total decode tokens\n", ti, generated);
    }
    llama_sampler_free(greedy);
    fs::path out(output_path), tmp = output_path + ".tmp"; fs::remove_all(tmp); fs::create_directories(tmp);
    const size_t T = cap.is_decode.size();
    if (capture_enabled) {
        cap.experts.resize(T*cap.n_layers*cap.k); cap.gates.resize(T*cap.n_layers*cap.k);
        for (int il = 0; il < cap.n_layers; ++il) {
            if (cap.expert_layers[il].size() != T*cap.k || cap.gate_layers[il].size() != T*cap.k)
                throw std::runtime_error("routing callback token count mismatch at layer " + std::to_string(il) +
                    ": experts=" + std::to_string(cap.expert_layers[il].size()/cap.k) +
                    " gates=" + std::to_string(cap.gate_layers[il].size()/cap.k) + " expected=" + std::to_string(T));
            for (size_t t = 0; t < T; ++t) for (int j = 0; j < cap.k; ++j) {
                size_t dst = (t*cap.n_layers + il)*cap.k, src = t*cap.k;
                cap.experts[dst + j] = cap.expert_layers[il][src + j];
                cap.gates[dst + j] = cap.gate_layers[il][src + j];
            }
        }
        write_npy(tmp/"experts.npy", "<u2", {T,(size_t)cap.n_layers,(size_t)cap.k}, cap.experts);
        write_npy(tmp/"gates.npy", "<f2", {T,(size_t)cap.n_layers,(size_t)cap.k}, cap.gates);
        write_npy(tmp/"is_decode.npy", "|b1", {T}, cap.is_decode);
        write_npy(tmp/"turn_id.npy", "<i4", {T}, cap.turn_id);
        write_npy(tmp/"session_id.npy", "<i4", {T}, cap.session_id);
    }
    common_chat_templates_inputs final_in; final_in.messages = messages; final_in.add_generation_prompt = false;
    final_in.use_jinja = params.use_jinja;
    std::ofstream(tmp/"generated.txt") << common_chat_templates_apply(tmpls.get(), final_in).prompt;
    std::ofstream metrics(tmp/"metrics.json");
    metrics << "{\n  \"decode_tokens\": " << generated << ",\n  \"decode_seconds\": " << decode_seconds
            << ",\n  \"decode_tokens_per_second\": " << generated/decode_seconds << "\n}\n"; metrics.close();
    std::ofstream manifest(tmp/(capture_enabled ? "manifest.json" : "benchmark.json"));
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    manifest << "{\n  \"model\": \"" << params.model.path << "\",\n  \"arch\": \"qwen4exp\",\n"
             << "  \"n_layers\": " << cap.n_layers << ",\n  \"n_experts\": " << n_experts << ",\n  \"k\": " << cap.k
             << ",\n  \"llama_cpp_commit\": \"" << llama_commit() << "\",\n  \"created\": \""
             << std::put_time(std::gmtime(&now), "%FT%TZ") << "\"\n}\n"; manifest.close();

    llama_log_set(log_cb, nullptr); init.reset(); common_log_flush(common_log_main());
    std::ofstream(tmp/"cache-stats.txt") << cache_log;
    fs::remove_all(out); fs::rename(tmp, out);
    std::fprintf(stderr, "%s %zu tokens (%zu decode), %.2f decode tok/s to %s\n",
        capture_enabled ? "logged" : "benchmarked", T, generated, generated/decode_seconds, output_path.c_str());
    llama_backend_free(); return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "llama-routelog: %s\n", e.what()); return 1;
}
