#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <filesystem>
#include <openvino/openvino.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>
#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/genai/generation_handle.hpp>
#include "openvino/genai/text_streamer.hpp"
#include "load_image.hpp"

#ifdef _WIN32
#    include <fcntl.h>
#    include <io.h>
#    include <stdlib.h>
#    include <windows.h>
#    include <psapi.h>

#    include <codecvt>
#    pragma comment(lib, "psapi.lib")  // PrintMemoryInfo
#    include <stdio.h>

#    include "processthreadsapi.h"
#endif

/**
 * @brief Read image from file path and convert to OpenVINO Tensor
 * @param path Image file path
 * @return OpenVINO Tensor containing image data
 */

struct PromptWithImage {
    std::string prompt;
    std::string image_path;
};

struct GenaiArgs {
    std::string model_path;
    std::string device = "GPU";
    std::vector<PromptWithImage> prompts_with_images;
};

static void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
        << "\n"
        << "options:\n"
        << "  -h, --help                         show this help message and exit\n"
        << "  -m, --model PATH                   model path\n"
        << "  -d, --device DEVICE                device name (default: GPU)\n"
        << "  --prompt_with_image PROMPT PATH    add one prompt-image request; repeat this option for multiple requests\n";
}

static GenaiArgs parse_args(const std::vector<std::string>& argv) {
    GenaiArgs args;

    for (size_t i = 1; i < argv.size(); ++i) {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        else if (arg == "-m" || arg == "--model") {
            if (i + 1 >= argv.size()) {
                throw std::runtime_error("Missing value for --model.");
            }
            args.model_path = argv[++i];
        }
        else if (arg == "-d" || arg == "--device") {
            if (i + 1 >= argv.size()) {
                throw std::runtime_error("Missing value for --device.");
            }
            args.device = argv[++i];
        }
        else if (arg == "--prompt_with_image") {
            if (i + 2 >= argv.size()) {
                throw std::runtime_error("--prompt_with_image requires PROMPT and PATH.");
            }
            args.prompts_with_images.push_back(PromptWithImage{ argv[++i], argv[++i] });
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (args.model_path.empty()) {
        throw std::runtime_error("Model path is required. Use --model PATH.");
    }

    if (args.prompts_with_images.empty()) {
        throw std::runtime_error(
            "At least one prompt-image pair is required. Use --prompt_with_image PROMPT PATH.");
    }

    return args;
}

static GenaiArgs parse_args(int argc, char** argv) {
    std::vector<std::string> argv_vec;
    argv_vec.reserve(argc);

#ifdef _WIN32
    LPWSTR* wargs = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!wargs) {
        throw std::runtime_error("Failed to parse command line arguments.");
    }

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    for (int i = 0; i < argc; ++i) {
        argv_vec.emplace_back(converter.to_bytes(wargs[i]));
    }

    LocalFree(wargs);
#else
    for (int i = 0; i < argc; ++i) {
        argv_vec.emplace_back(argv[i]);
    }
#endif

    return parse_args(argv_vec);
}

int main(int argc, char** argv) {
    try {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        _setmode(_fileno(stdin), _O_WTEXT);
#endif
        GenaiArgs genai_args = parse_args(argc, argv);

        if (!std::filesystem::exists(genai_args.model_path)) {
            throw std::runtime_error("Model path does not exist: " + genai_args.model_path);
        }

        for (const auto& request : genai_args.prompts_with_images) {
            if (!std::filesystem::exists(request.image_path)) {
                throw std::runtime_error("Image path does not exist: " + request.image_path);
            }
        }

        // Configure scheduler settings
        ov::genai::SchedulerConfig scheduler_config;
        scheduler_config.max_num_batched_tokens = 256;

        // Model and device configuration
        std::filesystem::path model_path = genai_args.model_path;
        std::string device = genai_args.device;
        ov::AnyMap properties;
        properties = { ov::cache_dir(std::format("{}_cache", device)) };

        // Initialize continuous batching pipeline
        ov::genai::ContinuousBatchingPipeline pipe(model_path, scheduler_config, device, properties);

        // Configure text generation parameters
        ov::genai::GenerationConfig generation_config;
        generation_config.max_new_tokens = 100;
        generation_config.do_sample = false;

        // Get tokenizer from pipeline
        auto tokenizer = pipe.get_tokenizer();

        // Storage for streamers and generation handles
        std::vector<std::unique_ptr<ov::genai::TextStreamer>> streamers;
        std::vector<ov::genai::GenerationHandle> generation_handles;
        streamers.reserve(genai_args.prompts_with_images.size());
        generation_handles.reserve(genai_args.prompts_with_images.size());

        // Add requests to pipeline for each prompt-image pair
        for (size_t request_id = 0; request_id < genai_args.prompts_with_images.size(); ++request_id) {
            const auto& request = genai_args.prompts_with_images[request_id];

            // Create custom text streamer for this request
            auto streamer = std::make_unique<ov::genai::TextStreamer>(
                tokenizer,
                [request_id](std::string text) -> ov::genai::CallbackTypeVariant {
                    std::cout << "\nRequest " << request_id << ": " << text << std::flush;
                    return ov::genai::StreamingStatus::RUNNING;
                }
            );
            streamers.push_back(std::move(streamer));

            // Load and prepare image tensor
            std::vector<ov::Tensor> images = { utils::load_image(request.image_path) };

            // Add request to pipeline
            auto handle = pipe.add_request(request_id, request.prompt, images, generation_config);
            generation_handles.push_back(std::move(handle));
        }

        // Process requests until all are completed
        while (pipe.has_non_finished_requests()) {
            // Execute one step of generation
            pipe.step();

            // Check outputs for each request
            for (size_t request_id = 0; request_id < generation_handles.size(); ++request_id) {
                auto& handle = generation_handles[request_id];

                // Read available outputs if ready
                if (handle->can_read()) {
                    auto outputs = handle->read();

                    // Process generated tokens
                    for (const auto& [key, output] : outputs) {
                        (void)key;
                        streamers[request_id]->write(output.generated_ids);
                    }
                }
            }
        }

        // Finalize all streamers
        for (auto& streamer : streamers) {
            streamer->end();
        }

        std::cout << "\nAll requests completed successfully!" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
