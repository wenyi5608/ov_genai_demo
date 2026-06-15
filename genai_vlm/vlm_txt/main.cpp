// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <algorithm>
#include <filesystem>
#include <format>
#include <openvino/genai/visual_language/pipeline.hpp>
#include "openvino/genai/text_streamer.hpp"
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
    std::string lora_path = "adapter_model.safetensors";
    std::string device = "GPU";
    float lora_alpha = 0.5;
    bool enable_lora = false;
    int output_fixed_len = 0;
    std::string test_mode = "memory";     
};

static void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "options:\n"
              << "  -h, --help              show this help message and exit\n"
              << "  -m, --model PATH        llm model path \n"
              << "  -lora_adapter PATH      lora adapter model file (default: adapter_model.safetensors)\n"
              << "  -lora_alpha N           lora_alpha (default: 0.5)\n"
              << "  -d, --device            Device (default: GPU)\n"
              << "  --test_mode             test mode (default: memory)\n";
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
        } else if (arg == "-lora_adapter") {
            args.lora_path = argv[++i];
        } else if (arg == "-lora_alpha") {
            args.lora_alpha = std::stof(argv[++i]);
        } else if (arg == "-d" || arg == "--device") {
            args.device = argv[++i];
        } else if (arg == "--test_mode") {
            args.test_mode = argv[++i];
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

enum class TestMode {
    invalid = 0,
    performance = 1,
    memory = 2,
    empty_lora_performance = 3,
    empty_lora_memory = 4,
    infer_with_lora_performance = 5,
    infer_with_lora_memory = 6,
};

TestMode parse_args(const std::string& mode) {
    if (mode == "performance") {
        return TestMode::performance;
    } else if (mode == "memory") {
        return TestMode::memory;
    } else if (mode == "empty_lora_performance") {
        return TestMode::empty_lora_performance;
    } else if (mode == "empty_lora_memory") {
        return TestMode::empty_lora_memory;
    } else if (mode == "infer_with_lora_performance") {
        return TestMode::infer_with_lora_performance;
    } else if (mode == "infer_with_lora_memory") {
        return TestMode::infer_with_lora_memory;
    } else {
        throw std::runtime_error("Invalid test mode.\n");
    }
    return TestMode::invalid;
}

namespace fs = std::filesystem;

int main(int argc, char* argv[]) try {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    _setmode(_fileno(stdin), _O_WTEXT);
#endif

    GenaiArgs genai_args = parse_args(argc, argv);

    std::filesystem::path models_path = genai_args.llm_model_path;
    std::filesystem::path adapter_path = genai_args.lora_path;

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

    using namespace ov::genai;
    std::cout << ov::get_openvino_version() << std::endl;
 
    auto start_time = std::chrono::steady_clock::now();
    Adapter adapter;

    if (!(test_mode == TestMode::memory) && !(test_mode == TestMode::performance)) {
        adapter = Adapter(adapter_path);
    }
 
    auto stop_time = std::chrono::steady_clock::now();

    size_t load_time = PerfMetrics::get_microsec(stop_time - start_time);
    std::cout << "lora load time " << load_time / 1000 << " ms" << std::endl;

    ov::AnyMap mp; 
    if (test_mode == TestMode::memory || test_mode == TestMode::performance) {
        mp = {{"ATTENTION_BACKEND", "PA"},
              ov::device::properties(device, ov::cache_dir(std::format("{}_cache", device)))};
    } else {
        mp = {{"ATTENTION_BACKEND", "PA"},
              ov::device::properties(device, ov::cache_dir(std::format("{}_cache", device))),
              adapters(adapter)};
    }

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.enable_prefix_caching = false;
    //scheduler_config.max_num_batched_tokens = 4096;
    //mp.emplace(ov::genai::scheduler_config.name(), scheduler_config);
    //mp.emplace(ov::hint::kv_cache_precision.name(), ov::element::i4);

    ov::genai::VLMPipeline pipe(models_path, device, mp);
    ov::InferRequest tokenizer = core.compile_model(models_path / "openvino_tokenizer.xml", "CPU").create_infer_request();
    ov::InferRequest detokenizer = core.compile_model(models_path / "openvino_detokenizer.xml", "CPU").create_infer_request();

    int idx = 0;
    // only used in memory test , test the memory usage after the first inference
    auto streamer = [](std::string subword) {
#if defined(WIN32)
       DebugMemoryInfo("First token ");
#endif
        return ov::genai::StreamingStatus::STOP;
    };

    // input length, output length, first time, other time
    std::vector<std::tuple<size_t, size_t, float, float>> perf_records;
    ov::genai::GenerationConfig config;

    config.apply_chat_template = FALSE;

    if (test_mode == TestMode::memory 
        || test_mode == TestMode::empty_lora_memory
        || test_mode == TestMode::infer_with_lora_memory) {
        config.max_new_tokens = 1;  // streamer may inpact the performance test, only infer first token for the memory test
    } else {
        config.max_new_tokens = 512; // perfromance test
    }

    if (test_mode == TestMode::empty_lora_memory
        || test_mode == TestMode::empty_lora_performance) {
        config.adapters = ov::genai::AdapterConfig{};
    } else if (test_mode == TestMode::infer_with_lora_memory
        || test_mode == TestMode::infer_with_lora_performance) {
        config.adapters = ov::genai::AdapterConfig{adapter, 0.5};
    }

    size_t prompt_idx = 0;
    ov::genai::VLMDecodedResults vlm_res;

    size_t input_ids_len[] = { 1024, 2 * 1024, 4 * 1024,  8 * 1024, 16 * 1024 }; // , 32 * 1024, 48 * 1024, 64 * 1024, 80 * 1024, 96 * 1024 };

    std::ifstream file(prompts_path_test);
    std::getline(file, str, '\0');

    auto [full_input_ids, attention_mask] = tokenize(tokenizer, std::string(str));

    const int64_t* full_input_ids_data = full_input_ids.data<const int64_t>();
    const size_t full_input_ids_len = full_input_ids.get_size();

    std::cout << "inputs total len " << full_input_ids_len << std::endl;

    for (size_t requested_input_ids_len : input_ids_len) {
        if (requested_input_ids_len > full_input_ids_len) {
            std::cout << "Skip requested input_ids length " << requested_input_ids_len
                      << ", available input_ids length is " << full_input_ids_len << std::endl;
            continue;
        }

        std::vector<int64_t> sliced_input_ids(full_input_ids_data+prompt_idx*10, full_input_ids_data + requested_input_ids_len+prompt_idx*10);
        std::string prompt_by_input_ids = detokenize(detokenizer, sliced_input_ids);

        if (test_mode == TestMode::memory || test_mode == TestMode::infer_with_lora_memory) {
            vlm_res = pipe.generate(prompt_by_input_ids, ov::genai::generation_config(config),
                ov::genai::streamer(streamer));
        }
        else {
            vlm_res = pipe.generate(prompt_by_input_ids, ov::genai::generation_config(config));
        }

        ov::genai::PerfMetrics metrics = vlm_res.perf_metrics;
        size_t input_tokens_len = metrics.get_num_input_tokens();
        size_t num_generated_tokens = metrics.get_num_generated_tokens();

        if (!prompt_idx) {
            std::cout << "Compile LLM model took " << metrics.get_load_time() << " ms" << std::endl;
        }

        std::cout << vlm_res.texts[0] << std::endl;
        prompt_idx++;

        perf_records.emplace_back(input_tokens_len,
            num_generated_tokens,
            metrics.get_ttft().mean,
            metrics.get_tpot().mean);
    }

    if (test_mode == TestMode::performance || test_mode == TestMode::empty_lora_performance ||
        test_mode == TestMode::infer_with_lora_performance) {
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
