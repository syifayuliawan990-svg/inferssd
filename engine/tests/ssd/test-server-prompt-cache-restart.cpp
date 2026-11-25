// SPDX-License-Identifier: Apache-2.0
#include "server-context.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

static constexpr int prefix_tokens = 96;
static constexpr int prompt_tokens = 384;
static constexpr int vocab_size = 259;
static constexpr int width = 32;
static constexpr llama_token base_token = 3 + 'A';
static constexpr llama_token adapted_token = 3 + 'B';

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Uniform attention makes the first output depend on values stored during prefill.
// The logits choose A for the base model and B with the active value adapter.
static void write_model(const fs::path & path) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(ggml_init({1u << 20, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> model(gguf_init_empty(), gguf_free);
    require(weights && model, "could not create prompt-cache model fixture");
    gguf_set_val_str(model.get(), "general.architecture", "llama");
    gguf_set_val_str(model.get(), "tokenizer.ggml.model", "llama");
    gguf_set_val_u32(model.get(), "llama.context_length", 512);
    gguf_set_val_u32(model.get(), "llama.embedding_length", width);
    gguf_set_val_u32(model.get(), "llama.block_count", 1);
    gguf_set_val_u32(model.get(), "llama.feed_forward_length", width);
    gguf_set_val_u32(model.get(), "llama.attention.head_count", 1);
    gguf_set_val_f32(model.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(model.get(), "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_u32(model.get(), "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(model.get(), "tokenizer.ggml.eos_token_id", 2);

    std::vector<std::string> words = { "<unk>", "<s>", "</s>" };
    for (unsigned i = 0; i < 256; ++i) {
        char word[7];
        std::snprintf(word, sizeof(word), "<0x%02X>", i);
        words.emplace_back(word);
    }
    std::vector<const char *> tokens;
    for (const auto & word : words) {
        tokens.push_back(word.c_str());
    }
    std::vector<int32_t> types(vocab_size, 6);
    types[0] = 2;
    types[1] = types[2] = 3;
    const std::vector<float> scores(vocab_size, 0.0f);
    gguf_set_arr_str(model.get(), "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    gguf_set_arr_data(model.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_arr_data(model.get(), "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());

    const auto add = [&](const char * name, int rows = 0, float value = 0.0f) {
        auto * tensor = rows ? ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, width, rows) :
                               ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, width);
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        std::fill(values, values + ggml_nelements(tensor), value);
        gguf_add_tensor(model.get(), tensor);
        return values;
    };
    auto * embeddings = add("token_embd.weight", vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        embeddings[i * width] = 1.0f;
    }
    add("output_norm.weight", 0, 1.0f);
    auto * output = add("output.weight", vocab_size);
    output[base_token * width] = 1.0f;
    output[adapted_token * width + 1] = 1.0f;
    add("blk.0.attn_norm.weight", 0, 1.0f);
    add("blk.0.attn_q.weight", width);
    add("blk.0.attn_k.weight", width);
    add("blk.0.attn_v.weight", width);
    add("blk.0.attn_output.weight", width)[width + 1] = 1.0f;
    add("blk.0.ffn_norm.weight", 0, 1.0f);
    add("blk.0.ffn_gate.weight", width);
    add("blk.0.ffn_down.weight", width);
    add("blk.0.ffn_up.weight", width);
    require(gguf_write_to_file(model.get(), path.string().c_str(), false), "could not write prompt-cache model fixture");
}

// A rank-one update sends the first input component to the second value component.
static void write_adapter(const fs::path & path, float value) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(ggml_init({1u << 16, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> adapter(gguf_init_empty(), gguf_free);
    require(weights && adapter, "could not create LoRA fixture");
    gguf_set_val_str(adapter.get(), "general.type", "adapter");
    gguf_set_val_str(adapter.get(), "general.architecture", "llama");
    gguf_set_val_str(adapter.get(), "adapter.type", "lora");
    gguf_set_val_f32(adapter.get(), "adapter.lora.alpha", 1.0f);

    const auto add = [&](const char * name, int columns, int rows, int component, float value) {
        auto * tensor = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, columns, rows);
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        std::fill(values, values + ggml_nelements(tensor), 0.0f);
        values[component] = value;
        gguf_add_tensor(adapter.get(), tensor);
    };
    add("blk.0.attn_v.weight.lora_a", width, 1, 0, 1.0f);
    add("blk.0.attn_v.weight.lora_b", 1, width, 1, value);
    require(gguf_write_to_file(adapter.get(), path.string().c_str(), false), "could not write LoRA fixture");
}

static bool has_prompt_file(const fs::path & cache) {
    for (const auto & item : fs::directory_iterator(cache)) {
        if (item.path().extension() == ".prompt") {
            return true;
        }
    }
    return false;
}

// If the server got ahead of the reader, pause before computing the first batch
// after the file appears. The reader can then queue cancellation before the
// server gets through the rest of the prompt or generates an answer.
struct suffix_gate {
    fs::path cache;
    std::mutex mutex;
    std::condition_variable condition;
    bool released = false;

    explicit suffix_gate(fs::path cache) : cache(std::move(cache)) {}

    static bool observe(ggml_tensor *, bool ask, void * user_data) {
        if (ask) {
            auto & gate = *static_cast<suffix_gate *>(user_data);
            std::unique_lock<std::mutex> lock(gate.mutex);
            if (!gate.released && has_prompt_file(gate.cache)) {
                gate.condition.wait(lock, [&] { return gate.released; });
            }
        }
        return false;
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        condition.notify_all();
    }
};

struct server_run {
    server_context server;
    suffix_gate * gate;
    std::thread worker;

    explicit server_run(common_params params, suffix_gate * gate = nullptr) : gate(gate) {
        if (gate) {
            params.cb_eval = suffix_gate::observe;
            params.cb_eval_user_data = gate;
        }
        require(server.load_model(params), "could not load prompt-cache model fixture");
        worker = std::thread([this] { server.start_loop(); });
    }

    void stop() {
        if (gate) {
            gate->release();
        }
        if (worker.joinable()) {
            server.terminate();
            worker.join();
        }
    }

    ~server_run() {
        stop();
    }
};

static server_task_result_ptr receive(server_response_reader & reader) {
    const auto deadline = std::chrono::steady_clock::now() + 15s;
    auto result = reader.next([&] { return std::chrono::steady_clock::now() >= deadline; });
    require(result && !result->is_error(), "prompt-cache request failed");
    return result;
}

static void post_prompt(server_response_reader & reader, llama_token suffix, std::optional<float> scale, bool progress,
        int id_slot = -1) {
    llama_tokens prompt(prompt_tokens, suffix);
    std::fill_n(prompt.begin(), prefix_tokens, base_token);
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id = reader.get_new_id();
    task.id_slot = id_slot;
    task.tokens = server_tokens(prompt, false);
    task.n_shared_prefix = prefix_tokens;
    task.params.n_predict = 1;
    task.params.stream = progress;
    task.params.return_progress = progress;
    task.params.return_tokens = true;
    task.params.sampling.temp = 0;
    if (scale) {
        task.params.lora[0] = *scale;
    }
    reader.post_task(std::move(task));
}

static void save_canceled_prefix(const common_params & params, std::optional<float> scale = {}) {
    const fs::path cache = params.cache_disk;
    require(fs::create_directory(cache), "could not create prompt-cache directory");
    suffix_gate gate(cache);
    server_run first(params, &gate);
    auto reader = first.server.get_response_reader();
    post_prompt(reader, 3 + 'x', scale, true);
    // The server first acknowledges the request, before decoding any tokens.
    receive(reader);
    const auto result = receive(reader);
    const auto * progress = dynamic_cast<const server_task_result_cmpl_partial *>(result.get());
    require(progress && progress->is_progress && progress->progress.cache == 0 &&
            progress->progress.processed == prefix_tokens, "first prefill batch did not end at the shared prefix");
    require(has_prompt_file(cache), "first successful prefix batch was not saved");

    reader.stop();
    gate.release();
    // This task runs after cancellation and before we destroy the server.
    auto metrics = first.server.get_response_reader();
    server_task request(SERVER_TASK_TYPE_METRICS);
    request.id = metrics.get_new_id();
    metrics.post_task(std::move(request));
    const auto receipt = receive(metrics);
    const auto * state = dynamic_cast<const server_task_result_metrics *>(receipt.get());
    require(state && state->n_idle_slots == 1 && state->n_tokens_predicted_total == 0,
            "the first request was not canceled during prefill");
}

struct completion {
    int cached;
    llama_token token;
};

static void require_completion(completion actual, int cached, llama_token token, const char * message) {
    if (actual.cached != cached || actual.token != token) {
        std::fprintf(stderr, "Cached tokens: %d (expected %d), first token: %d (expected %d)\n",
                     actual.cached, cached, actual.token, token);
        throw std::runtime_error(message);
    }
}

static completion first_token(server_context & server, std::optional<float> scale = {}, int id_slot = -1) {
    auto reader = server.get_response_reader();
    post_prompt(reader, 3 + 'y', scale, false, id_slot);
    const auto result = receive(reader);
    const auto * done = dynamic_cast<const server_task_result_cmpl_final *>(result.get());
    require(done && done->tokens.size() == 1, "the request did not complete with exactly one token");
    return {done->n_prompt_tokens_cache, done->tokens.front()};
}

static completion first_token(const common_params & params, std::optional<float> scale = {}) {
    server_run restarted(params);
    return first_token(restarted.server, scale);
}

static void slot_file(server_context & server, const fs::path & path, bool save) {
    auto reader = server.get_response_reader();
    server_task request(save ? SERVER_TASK_TYPE_SLOT_SAVE : SERVER_TASK_TYPE_SLOT_RESTORE);
    request.id = reader.get_new_id();
    request.slot_action = {0, path.filename().string(), path.string()};
    reader.post_task(std::move(request));
    const auto result = receive(reader);
    const auto * receipt = dynamic_cast<const server_task_result_slot_save_load *>(result.get());
    require(receipt && receipt->is_save == save && receipt->n_tokens == prompt_tokens && receipt->n_bytes > 0,
            "explicit slot operation did not contain the entire prompt");
}

static void test_explicit_slots(common_params params, const fs::path & work) {
    params.cache_disk.clear();
    params.cache_ram_mib = 4;
    params.slot_save_path = work.string();
    const auto base_path    = work / "base.slot";
    const auto adapted_path = work / "adapted.slot";
    const auto legacy_path  = work / "legacy.slot";

    auto with_base_only = params;
    with_base_only.lora_adapters.clear();
    {
        server_run original(with_base_only);
        require_completion(first_token(original.server), 0, base_token, "cold base request failed");
        slot_file(original.server, base_path, true);
    }
    {
        auto writer_params = params;
        writer_params.cache_ram_mib = 0;
        server_run original(writer_params);
        require_completion(first_token(original.server, 1.0f), 0, adapted_token, "cold adapted request failed");
        slot_file(original.server, adapted_path, true);
        original.stop();

        // The old explicit save contains native state and tokens with no adapter identity.
        llama_tokens input(prompt_tokens, 3 + 'y');
        std::fill_n(input.begin(), prefix_tokens, base_token);
        require(llama_state_seq_save_file(original.server.get_llama_context(), legacy_path.string().c_str(), 0,
                    input.data(), input.size()) > 0, "could not write legacy slot fixture");
    }

    for (const int id_slot : {-1, 0}) {
        {
            server_run restarted(with_base_only);
            slot_file(restarted.server, base_path, false);
            require_completion(first_token(restarted.server, {}, id_slot), prompt_tokens - 1, base_token,
                    "a compatible base slot must retain its prefix when selected automatically or explicitly");
        }
        {
            server_run restarted(params);
            slot_file(restarted.server, adapted_path, false);
            require_completion(first_token(restarted.server, 1.0f, id_slot), prompt_tokens - 1, adapted_token,
                    "a slot saved with matching adapter weights and scale must reuse its prompt");
        }
        {
            server_run restarted(params);
            slot_file(restarted.server, adapted_path, false);
            require_completion(first_token(restarted.server, {}, id_slot), 0, base_token,
                    "a base request must recompute an adapted slot when selected automatically or explicitly");
        }
    }

    // An old adapted file remains unsafe even when the new server loads no adapter.
    server_run restarted(with_base_only);
    {
        auto reader = restarted.server.get_response_reader();
        server_task request(SERVER_TASK_TYPE_SLOT_RESTORE);
        request.id = reader.get_new_id();
        request.slot_action = {0, legacy_path.filename().string(), legacy_path.string()};
        reader.post_task(std::move(request));
        const auto deadline = std::chrono::steady_clock::now() + 15s;
        const auto result = reader.next([&] { return std::chrono::steady_clock::now() >= deadline; });
        require(result && result->is_error(), "legacy slot state without adapter provenance must be rejected");
    }
    require_completion(first_token(restarted.server, {}, 0), 0, base_token,
            "a rejected legacy restore must not populate the pinned slot");
}

static void test_restart(const fs::path & work) {
    const fs::path model = work / "model.gguf";
    const fs::path cache = work / "cache";
    write_model(model);
    common_params params;
    params.model.path = model.string();
    params.n_ctx = 512;
    params.n_batch = params.n_ubatch = 128;
    params.n_parallel = 1;
    params.n_gpu_layers = 0;
    params.devices = { nullptr };
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 1;
    params.n_ctx_checkpoints = 0;
    params.cache_disk = cache.string();
    params.cache_disk_mib = 4;
    params.warmup = false;
    params.fit_params = false;

    save_canceled_prefix(params);
    require_completion(first_token(params), prefix_tokens, base_token,
            "restart did not restore the exact shared prefix for a different suffix");

    const fs::path adapter = work / "adapter.gguf";
    write_adapter(adapter, 4.0f);
    params.lora_adapters.push_back({adapter.string(), 0.0f, "", "", nullptr});
    test_explicit_slots(params, work);
    params.cache_disk.clear();
    const auto cold_base = first_token(params);
    const auto cold_adapted = first_token(params, 1.0f);
    const auto cold_low_scale = first_token(params, 0.03125f);
    require(cold_base.token == base_token && cold_adapted.token == adapted_token && cold_low_scale.token == base_token,
            "the fixture must distinguish the adapter and its scale in the first output token");

    params.cache_disk = cache.string();
    require_completion(first_token(params), prefix_tokens, cold_base.token,
            "loading a disabled adapter must keep base-model reuse");

    const fs::path adapted_cache = work / "adapted-cache";
    params.cache_disk = adapted_cache.string();
    save_canceled_prefix(params, 1.0f);

    require_completion(first_token(params), 0, cold_base.token,
            "a default-scale request must match a cold base run after an adapted prefix was saved");
    require_completion(first_token(params, 0.03125f), 0, cold_low_scale.token,
            "a different nonzero scale must match a cold run");
    require_completion(first_token(params, 1.0f), prefix_tokens, cold_adapted.token,
            "matching adapters and scales must restore the full prefix and match a cold run");

    // Keep the path, metadata, size, and timestamp while replacing the adapter weights.
    const auto original_mtime = fs::last_write_time(adapter);
    write_adapter(adapter, 0.125f);
    fs::last_write_time(adapter, original_mtime);
    params.cache_disk.clear();
    const auto cold_replacement = first_token(params, 1.0f);
    require(cold_replacement.token == base_token, "the replacement adapter must produce a different token");
    params.cache_disk = adapted_cache.string();
    require_completion(first_token(params, 1.0f), 0, cold_replacement.token,
            "different adapter weights with the same scale must match a cold run");
}

int main() {
    const auto work = fs::temp_directory_path() / ("inferssd-prefix-restart-" + std::to_string(std::random_device{}()));
    try {
        require(fs::create_directory(work), "could not create prompt-cache test directory");
        struct cleanup {
            const fs::path & path;
            ~cleanup() { std::error_code error; fs::remove_all(path, error); }
        } remove_fixture{work};
        ggml_backend_load_all();
        llama_backend_init();
        test_restart(work);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
