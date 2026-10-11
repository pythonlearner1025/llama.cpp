#pragma once

#include <cstdint>
#include <memory>
#include <string>

struct llama_context;
struct llama_model;
struct llama_ubatch;
class llm_graph_result;

// Opt-in full-router capture owned by a llama_context. The implementation is
// deliberately outside common/ so LLAMA_ROUTER_LOG_DIR works for every libllama
// consumer, including programs which do not use the common argument parser.
class llama_router_log {
public:
    llama_router_log(const std::string & dir, double max_gb, const llama_model & model);
    ~llama_router_log();

    bool capture(llama_context & ctx, const llm_graph_result & graph, const llama_ubatch & ubatch);
    void drain(); // scheduler must already be synchronized
    void flush();
    void log_request(int32_t seq_id, int64_t task_id, const char * request_id,
                     int32_t prompt_tokens, int64_t start_unix_ms, int64_t end_unix_ms);
    bool matches(const llama_model & model) const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};

// Contexts in one process share one writer for a directory (server target/draft
// contexts and benchmark repetitions can otherwise collide on manifest.json).
std::shared_ptr<llama_router_log> llama_router_log_get(
        const std::string & dir, double max_gb, const llama_model & model);
