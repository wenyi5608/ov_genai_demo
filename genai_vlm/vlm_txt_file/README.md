# GenAI VLMPipeline with Text File Input

This is an example that shows the performance and memory testing of a OpenVINO.GenAI VLM pipeline.

## Build and Run


```
<OpenVINO_GenAI_DIR>\setupvars.bat
cd genai_vlm\vlm_txt_file
mkdir build
cmake -S . -B build && cmake --build build --config Release
.\build\Release\genai_vlm.exe -m \path\to\ov_llm_model -prompt \path\to\prompt -d GPU --test_mode  performance
```
