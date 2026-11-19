// SPDX-License-Identifier: Apache-2.0
// A Responses client such as Codex sends reasoning.effort in OpenAI's words without reading /props.
// oaicompat_responses_fit_reasoning_effort runs it as what the chat template offers, so the request is served; the
// chat-completions field reasoning_effort still takes only the template's own values.
#include "server-chat.h"
#include "server-common.h"
#include "chat.h"

#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error(message); }
}

// The reasoning_effort that a Responses effort runs as on a template with this offer; empty when none is sent.
static std::string fitted(const std::optional<common_chat_reasoning> & reasoning, const std::string & effort) {
    json body = {{"reasoning_effort", effort}};
    oaicompat_responses_fit_reasoning_effort(body, reasoning);
    return json_value(body, "reasoning_effort", std::string());
}

static void test_fit() {
    struct fit {
        const char * effort;
        const char * runs_as;
    };
    struct offer {
        const char *                         name;
        std::optional<common_chat_reasoning> reasoning;
        std::vector<fit>                     fits;
    };
    // The offers of the supported models' templates, as /props reports them, and of a model that does not think
    const common_chat_reasoning qwen    { true,  {},                                  ""      };
    const common_chat_reasoning deepseek{ true,  { "low", "medium", "high", "max" },   "high"  };
    const common_chat_reasoning muse    { false, { "low", "medium", "high", "xhigh" }, "xhigh" };
    const common_chat_reasoning fixed   { false, {},                                  ""      };
    const std::vector<offer> offers = {
        { "a switch and no efforts (Qwen3.6)", qwen,
          { { "none", "none" }, { "minimal", "" }, { "low", "" }, { "medium", "" }, { "high", "" }, { "xhigh", "" } } },
        { "a switch and efforts (DeepSeek V4.1)", deepseek,
          { { "none", "none" }, { "low", "low" }, { "high", "high" }, { "max", "max" }, { "minimal", "" },
            { "xhigh", "" } } },
        { "efforts and no switch (Muse)", muse,
          { { "none", "low" }, { "low", "low" }, { "xhigh", "xhigh" }, { "minimal", "" }, { "max", "" } } },
        { "thinking with nothing to choose", fixed,
          { { "none", "" }, { "low", "" } } },
        { "no thinking", std::nullopt,
          { { "none", "" }, { "low", "" } } },
    };
    for (const auto & o : offers) {
        for (const auto & f : o.fits) {
            const std::string got = fitted(o.reasoning, f.effort);
            require(got == f.runs_as, std::string(o.name) + ": reasoning.effort '" + f.effort + "' ran as '" + got +
                                          "', expected '" + f.runs_as + "'");
        }
    }

    // Nothing to fit: no effort, a null one, and a value of another type, which the request is refused for
    const common_chat_reasoning switch_only{ true, {}, "" };
    for (const char * text : { R"({})", R"({"reasoning_effort": null})", R"({"reasoning_effort": 3})" }) {
        json body = json::parse(text);
        oaicompat_responses_fit_reasoning_effort(body, switch_only);
        require(body == json::parse(text), std::string("changed a request with nothing to fit: ") + text);
    }
}

// The route's own steps on a Responses request, with a template that thinks and that enable_thinking turns off.
static void test_responses_request() {
    const std::string source =
        "{%- for m in messages -%}<|{{ m.role }}|>"
        "{%- if m.reasoning_content -%}<think>{{ m.reasoning_content }}</think>{%- endif -%}"
        "{{ m.content }}<|end|>{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|assistant|>"
        "{%- if enable_thinking is defined and not enable_thinking -%}<think></think>{%- else -%}<think>{%- endif -%}"
        "{%- endif -%}";
    server_chat_params opt{};
    opt.use_jinja          = true;
    opt.prefill_assistant  = true;
    opt.reasoning_format   = COMMON_REASONING_FORMAT_DEEPSEEK;
    opt.force_pure_content = true; // the prompt alone is under test, not a parser for the model's output
    opt.tmpls              = common_chat_templates_init(/* model= */ nullptr, source);
    opt.reasoning          = common_chat_templates_reasoning(opt.tmpls.get());
    require(opt.reasoning && opt.reasoning->can_turn_off && opt.reasoning->efforts.empty(),
            "the fixture template must offer a switch and no efforts, as Qwen3.6's does");

    std::vector<raw_buffer> files;
    const auto prompt = [&](const char * effort) {
        json request = json::parse(R"({"model": "local", "input": "hi", "stream": true})");
        request["reasoning"] = {{"effort", effort}};
        json body = server_chat_convert_responses_to_chatcmpl(request);
        oaicompat_responses_fit_reasoning_effort(body, opt.reasoning);
        return oaicompat_chat_params_parse(body, opt, files).at("prompt").get<std::string>();
    };
    // An effort the template does not have is served, thinking as a request without one does
    for (const char * effort : { "minimal", "low", "medium", "high", "xhigh" }) {
        require(string_ends_with(prompt(effort), "<|assistant|><think>"),
                std::string("reasoning.effort '") + effort + "' must run with thinking on");
    }
    require(string_ends_with(prompt("none"), "<|assistant|><think></think>"),
            "reasoning.effort 'none' must turn thinking off");

    // The chat-completions field is not fitted: it takes what /props reports and nothing else
    json chat = json::parse(R"({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"})");
    bool refused = false;
    try {
        oaicompat_chat_params_parse(chat, opt, files);
    } catch (const std::invalid_argument &) {
        refused = true;
    }
    require(refused, "chat completions must refuse a reasoning_effort the template does not offer");
}

int main(int argc, char **) {
    try {
        require(argc == 1, "usage: test-server-reasoning-effort");
        test_fit();
        test_responses_request();
        std::puts("responses reasoning effort: PASS (no service started)");
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
