// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <filesystem>
#include <format>
#include "openvino/genai/llm_pipeline.hpp"
#include "utils.h"
#include "config.h" // path for prompts

#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <stdlib.h>
#include <psapi.h>
#pragma comment(lib,"psapi.lib") //PrintMemoryInfo
#include <stdio.h>
#include "processthreadsapi.h"
#endif

#include <openvino/openvino.hpp>


#ifdef WIN32
// To ensure correct resolution of symbols, add Psapi.lib to TARGETLIBS
// and compile with -DPSAPI_VERSION=1
static void DebugMemoryInfo(const char* header) {
    PROCESS_MEMORY_COUNTERS_EX2 pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        // The following printout corresponds to the value of Resource Memory, respectively
        printf("%s Commit \t\t\t=  0x%08X- %u (MB)\n", header, pmc.PrivateUsage, pmc.PrivateUsage / (1024 * 1024));
        printf("%s WorkingSetSize\t\t\t=  0x%08X- %u (MB)\n",
               header,
               pmc.WorkingSetSize,
               pmc.WorkingSetSize / (1024 * 1024));
        printf("%s PrivateWorkingSetSize\t\t\t=  0x%08X- %u (MB)\n",
               header,
               pmc.PrivateWorkingSetSize,
               pmc.PrivateWorkingSetSize / (1024 * 1024));
    }
}
#endif  //  WIN32

struct GenaiArgs {
    std::string llm_model_path = "";
    std::string device = "GPU";
    bool enable_xattention = FALSE;
    std::string test_mode = "performance";     
};

static void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "options:\n"
              << "  -h, --help              show this help message and exit\n"
              << "  -m, --model PATH        llm model path \n"
              << "  -d, --device            Device (default: GPU)\n"
              << "  --enable_xattention     XAttention ON/OFF (default: OFF)\n"
              << "  --test_mode             test mode (default: performance)\n";
}

static GenaiArgs parse_args(const std::vector<std::string>& argv) {
    GenaiArgs args;

    for (size_t i = 1; i < argv.size(); i++) {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (arg == "-m" || arg == "--model") {
            args.llm_model_path = argv[++i];
        } else if (arg == "-d" || arg == "--device") {
            args.device = argv[++i];
        } else if (arg == "--test_mode") {
            args.test_mode = argv[++i];
        } else if (arg == "--enable_xattention") {
            args.enable_xattention = TRUE;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return args;
}

static GenaiArgs parse_args(int argc, char** argv) {
    std::vector<std::string> argv_vec;
    argv_vec.reserve(argc);

#ifdef _WIN32
    LPWSTR* wargs = CommandLineToArgvW(GetCommandLineW(), &argc);

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(converter.to_bytes(wargs[i]));
    }

    LocalFree(wargs);
#else
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(argv[i]);
    }
#endif

    return parse_args(argv_vec);
}

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string&& prompt) {
    constexpr size_t BATCH_SIZE = 1;
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t>& tokens) {
    constexpr size_t BATCH_SIZE = 1;
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {BATCH_SIZE, tokens.size()}, tokens.data()});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

// The following reasons require TextStreamer to keep a cache of previous tokens:
// detokenizer removes starting ' '. For example detokenize(tokenize(" a")) == "a",
// but detokenize(tokenize("prefix a")) == "prefix a"
// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
struct TextStreamer {
    ov::InferRequest detokenizer;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;

    void put(int64_t token) {
        token_cache.push_back(token);
        std::string text = detokenize(detokenizer, token_cache);
        if (!text.empty() && '\n' == text.back()) {
            // Flush the cache after the new line symbol
            std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
            token_cache.clear();
            print_len = 0;
            return;
        }
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            // Don't print incomplete text
            return;
        }
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
        print_len = text.size();
    }

    void end() {
        std::string text = detokenize(detokenizer, token_cache);
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
        token_cache.clear();
        print_len = 0;
    }
};


enum class TestMode {
    invalid = 0,
    performance = 1,
    memory = 2,
};

TestMode parse_args(const std::string& mode) {
    if (mode == "performance") {
        return TestMode::performance;
    } else if (mode == "memory") {
        return TestMode::memory;
    } else {
        throw std::runtime_error("Invalid test mode.\n");
    }
    return TestMode::invalid;
}

int main(int argc, char* argv[]) try {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    _setmode(_fileno(stdin), _O_WTEXT);
#endif

    GenaiArgs genai_args = parse_args(argc, argv);

    std::filesystem::path models_path = genai_args.llm_model_path;

    TestMode test_mode = parse_args(std::string(genai_args.test_mode));
    std::cout << "test mode:" << (int)test_mode << std::endl;

    std::string device = genai_args.device;  // CPU can be used as well

    ov::Core core;

#ifdef _WIN32
    core.add_extension("openvino_tokenizers.dll");
#elif defined(__linux__)
    core.add_extension("libopenvino_tokenizers.so");
#endif

    std::filesystem::path prompts_path_test{CURRENT_SOURCE_DIR};
    prompts_path_test /= "A_Bid_for_Fortune.txt";

    std::string str;
    std::ifstream file(prompts_path_test);
    std::getline(file, str, '\0');

    std::string token_model_path = genai_args.llm_model_path;
    std::string token_model = token_model_path + "/openvino_tokenizer.xml";
    std::string detoken_model = token_model_path + "/openvino_detokenizer.xml";

    ov::InferRequest tokenizer = core.compile_model(token_model, "CPU").create_infer_request();
    ov::InferRequest detokenizer = core.compile_model(detoken_model, "CPU").create_infer_request();
    TextStreamer text_streamer{detokenizer};

    auto input_ids = tokenizer.get_tensor("input_ids");

    tokenize(tokenizer, str.c_str());

    std::cout << "input_ids length " << input_ids.get_size() << std::endl;

    using namespace ov::genai;
    std::cout << ov::get_openvino_version() << std::endl;
 
    ov::AnyMap mp; 
    mp = {ov::device::properties(device, ov::cache_dir(std::format("{}_cache", device)))};

    if (genai_args.enable_xattention) {
        ov::genai::SchedulerConfig scheduler_config;
        scheduler_config.enable_prefix_caching = false;
        scheduler_config.max_num_batched_tokens = 4096;
        scheduler_config.use_sparse_attention = true;
        scheduler_config.sparse_attention_config.mode = SparseAttentionMode::XATTENTION;
        scheduler_config.sparse_attention_config.xattention_threshold = 0.9;
        mp.emplace(ov::genai::scheduler_config.name(), scheduler_config);

        std::cout << "Sparse attention: XAttention " << std::endl;
    }

    mp.emplace(ov::hint::kv_cache_precision.name(), ov::element::f16);

    LLMPipeline pipe(models_path, device, mp);

    int idx = 0;
    // only used in memory test , test the memory usage after the first inference
    auto streamer = [](std::string subword) {
#ifdef WIN32
       DebugMemoryInfo("First token ");
#endif
        return ov::genai::StreamingStatus::STOP;
    };

    // input length, output length, first time, other time
    std::vector<std::tuple<size_t, size_t, float, float>> perf_records;
    ov::genai::GenerationConfig config;

    if (test_mode == TestMode::memory ) {
        config.max_new_tokens = 1;  // streamer may inpact the performance test, only infer first token for the memory test
    } else {
        config.max_new_tokens = 200; // perfromance test
    }

    size_t input_ids_len[] = { 1000, 3000, 8000, 10000, 15000, 20000, 25000, 30000};

    for (size_t i = 0; i < 6; i++) {
        ov::genai::EncodedResults enc_res;
        size_t input_id_len = std::min(input_ids_len[i], input_ids.get_size());

        ov::Tensor encode_inputs_id = ov::Tensor(ov::element::i64, {1, input_id_len}, &input_ids.data<int64_t>()[i*10]);

        if (test_mode == TestMode::memory) {
            enc_res = pipe.generate(encode_inputs_id, config, streamer);
        } else {
            enc_res = pipe.generate(encode_inputs_id, config);
        }

        ov::genai::PerfMetrics metrics = enc_res.perf_metrics;

        size_t input_tokens_len = metrics.get_num_input_tokens();
        size_t num_generated_tokens = metrics.get_num_generated_tokens();

        if (!i) {
            std::cout << "Compile LLM model took " << metrics.get_load_time() << " ms" << std::endl;
        }

        if (test_mode == TestMode::performance) {
            for (size_t idx = 0; idx < enc_res.tokens[0].size(); ++idx) {
                text_streamer.put(enc_res.tokens[0][idx]);
             }

            text_streamer.end();
        }

        perf_records.emplace_back(input_tokens_len,
                                  num_generated_tokens,
                                  metrics.get_ttft().mean,
                                  metrics.get_tpot().mean);

    }

    if (test_mode == TestMode::performance) {
        std::cout << "input id, input token len, out token len, first token time, average time" << std::endl;
        size_t index = 0;
        for (auto i : perf_records) {
            std::cout << index << ", " << std::get<0>(i) << ", " << std::get<1>(i) << ", " << std::get<2>(i) << ", "
                      << std::get<3>(i) << std::endl;
            index++;
        }
    }

    perf_records.clear();

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
