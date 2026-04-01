# Utilizing XAttention with GenAI LLMPipeline

This is an example that shows the performance and memory testing of a OpenVINO.GenAI LLM pipeline integrated with XAttention.

## Build and Run


```
<OpenVINO_GenAI_DIR>\setupvars.bat
cd genai_sparse_attention
mkdir build
cmake -S . -B build && cmake --build build --config Release
.\build\Release\genai_sparse_attention.exe -m \path\to\ov_llm_model  -d GPU ---enable_xattention
```
