# GenAI VLMPipeline

This is an example that shows the performance and memory testing of a OpenVINO.GenAI VLM pipeline.

## Build and Run
Windows

Download and Install VS2022, Cmake:

VS2022: Install latest [Visual Studio 2022 Community](https://visualstudio.microsoft.com/zh-hans/downloads/) and Install C and C++ support in Visual Studio.
Cmake: If Cmake not installed in the terminal Command Prompt, please download and install Cmake or use the terminal Developer Command Prompt for VS 2022 instead.

openvino_genai can be download from https://storage.openvinotoolkit.org/repositories/openvino_genai/packages

```
<OpenVINO_GenAI_DIR>\setupvars.bat
cd vlm_multi_batch
mkdir build
cmake -S . -B build && cmake --build build --config Release
.\build\Release\genai_vlm_multi_batch.exe -m  \path\to\model_path -d GPU --prompt_with_image "prompt1" "\path\to\image1" --prompt_with_image "prompt2" "\path\to\Image.jpg"
```
