# GenAI LLMPipeline

This is an example that shows the performance and memory testing of a OpenVINO.GenAI LLM pipeline.

## Build and Run
Windows

Download and Install VS2022, Cmake:

VS2022: Install latest [Visual Studio 2022 Community](https://visualstudio.microsoft.com/zh-hans/downloads/) and Install C and C++ support in Visual Studio.
Cmake: If Cmake not installed in the terminal Command Prompt, please download and install Cmake or use the terminal Developer Command Prompt for VS 2022 instead.

openvino_genai can be download from https://storage.openvinotoolkit.org/repositories/openvino_genai/packages

```
<OpenVINO_GenAI_DIR>\setupvars.bat
cd genai_llm
mkdir build
cmake -S . -B build && cmake --build build --config Release
.\build\Release\genai_llm.exe  "\\path\\to\\Qwen2.5-1.5B" "infer_with_memory"
.\build\Release\genai_llm.exe  "\\path\\to\\Qwen2.5-1.5B" "infer_with_performance"
```
