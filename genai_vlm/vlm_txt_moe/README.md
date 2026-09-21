# GenAI VLMPipeline with a Mixture-of-Experts (MoE) Model  in in text-only input scenarios

Benchmarks inference performance and memory usage of the OpenVINO GenAI VLM pipeline with an MoE-based model.

## Build and Run


```
<OpenVINO_GenAI_DIR>\setupvars.bat
cd genai_vlm\vlm_txt_moe
mkdir build
cmake -S . -B build && cmake --build build --config Release
.\build\Release\genai_vlm.exe -m \path\to\ov_llm_model -offload_ratio 50 -d GPU --test_mode  performance
```
