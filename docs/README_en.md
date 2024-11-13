# Digital Human Conversation Demo  
A real-time voice interactive digital human supporting both end-to-end voice solutions (GLM-4-Voice - THG) and cascaded solutions (ASR-LLM-TTS-THG). Customizable avatar and voice, with voice cloning capability and first-package latency as low as 3 seconds.

Online Demo: [https://www.modelscope.cn/studios/AI-ModelScope/video_chat](https://www.modelscope.cn/studios/AI-ModelScope/video_chat)

For detailed technical introduction, please refer to [this article](https://mp.weixin.qq.com/s/jpoB8O2IyjhXeAWNWnAj7A).

**Simplified Chinese** | [**English**](./docs/README_en.md)

## TODO  
- [x] Add voice cloning functionality to the TTS module  
- [x] Add `edge-tts` support to the TTS module  
- [x] Add local inference for the Qwen LLM module  
- [x] Support GLM-4-Voice and provide both ASR-LLM-TTS-THG and MLLM-THG generation methods  
- [ ] Integrate vLLM for inference acceleration with GLM-4-Voice  
- [ ] Integrate [gradio-webrtc](https://github.com/freddyaboulton/gradio-webrtc) (pending support for audio-video synchronization) to improve video stream stability
 
## Technical Stack  
* **ASR (Automatic Speech Recognition):** [FunASR](https://github.com/modelscope/FunASR)  
* **LLM (Large Language Model):** [Qwen](https://github.com/QwenLM/Qwen)  
* **End-to-end MLLM (Multimodal Large Language Model):** [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice/tree/main)  
* **TTS (Text to Speech):** [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), [edge-tts](https://github.com/rany2/edge-tts)  
* **THG (Talking Head Generation):** [MuseTalk](https://github.com/TMElyralab/MuseTalk/tree/main)  

## Local Deployment  
### 0. GPU Memory Requirements  
* **Cascaded Solution (ASR-LLM-TTS-THG):** ~8G, first-package latency ~3s (single A100 GPU).  
* **End-to-end Voice Solution (MLLM-THG):** ~20G, first-package latency ~7s (single A100 GPU).  
Developers who do not require the end-to-end MLLM solution can use the `cascade_only` branch.
```bash
$ git checkout cascade_only
```

### 1. Environment Setup  
* Ubuntu 22.04  
* Python 3.10  
* CUDA 12.2  
* PyTorch 2.3.0
```bash
$ git lfs install
$ git clone https://www.modelscope.cn/studios/AI-ModelScope/video_chat.git
$ conda create -n metahuman python=3.10
$ conda activate metahuman
$ cd video_chat
$ pip install -r requirement.txt
```

### 2. Weight Downloads  
#### 2.1 Via Creative Space (Recommended)  
The Creative Space repository has `git lfs` tracking for weight files. 

If cloned via `git clone https://www.modelscope.cn/studios/AI-ModelScope/video_chat.git`, no additional setup is needed.  

#### 2.2 Manual Downloads  
##### 2.2.1 MuseTalk  
Refer to [this link](https://github.com/TMElyralab/MuseTalk/blob/main/README.md#download-weights).  
Directory structure:  
```plaintext
./weights/
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── musetalk
│   ├── musetalk.json
│   └── pytorch_model.bin
├── sd-vae-ft-mse
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    └── tiny.pt
```

##### 2.2.2 GPT-SoVITS  
Refer to [this link](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/cn/README.md#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B).  

##### 2.2.3 GLM-4-Voice  
Add the following code in `app.py` to download weights:  
```python
from modelscope import snapshot_download
snapshot_download('ZhipuAI/glm-4-voice-tokenizer', cache_dir='./weights')
snapshot_download('ZhipuAI/glm-4-voice-decoder', cache_dir='./weights')
snapshot_download('ZhipuAI/glm-4-voice-9b', cache_dir='./weights')
```

### 3. Additional Configuration  
LLM and TTS modules offer multiple inference options:  

#### 3.1 Using API-KEY (Default)  
For LLM and TTS modules, if local machine performance is limited, you can use Alibaba Cloud's Qwen API and CosyVoice API. Configure the API-KEY in `app.py` (line 14):  
```python
os.environ["DASHSCOPE_API_KEY"] = "INPUT YOUR API-KEY HERE"
```

#### 3.2 Without API-KEY  
If not using API-KEY, update the relevant code as follows:  

##### 3.2.1 LLM Module  
The `src/llm.py` file provides `Qwen` and `Qwen_API` classes for local inference and API calls respectively. Options for local inference:  
1. Use `Qwen` for local inference.  
2. Use `vLLM` for accelerated inference with `Qwen_API(api_key="EMPTY", base_url="http://localhost:8000/v1")`. Installation:
```bash
$ git clone https://github.com/vllm-project/vllm.git
$ cd vllm
$ python use_existing_torch.py
$ pip install -r requirements-build.txt
$ pip install -e . --no-build-isolation
```
Refer to [this guide](https://qwen.readthedocs.io/zh-cn/latest/getting_started/quickstart.html#vllm-for-deployment) for deployment.

##### 3.2.2 TTS Module  
The `src/tts.py` file provides `GPT_SoVits_TTS` and `CosyVoice_API` for local inference and API calls. Use `Edge_TTS` for free TTS services.  

### 4. Start the Service  
$ python app.py

### 5. Customize Digital Humans (Optional)  
#### 5.1 Custom Avatar  
1. Add the recorded avatar video in `/data/video/`.  
2. Modify the `avatar_list` in the `Muse_Talk` class in `/src/thg.py` to include `(avatar_name, bbox_shift)`. Refer to [this link](https://github.com/TMElyralab/MuseTalk?tab=readme-ov-file#use-of-bbox_shift-to-have-adjustable-results) for details on `bbox_shift`.  
3. Add the avatar name in the `avatar_name` option in Gradio in `app.py`, restart the service, and wait for initialization to complete.

#### 5.2 Custom Voice  
`GPT-SoVits` supports custom voice cloning. To add a voice permanently:  
1. Add reference audio (3-10s, named as `x.wav`) to `/data/audio/`.  
2. Add the voice name (format: `x (GPT-SoVits)`) in the `avatar_voice` option in Gradio in `app.py` and restart the service.  
3. Set TTS option to `GPT-SoVits` and start interacting.

### 6. Known Issues  
1. Missing resources: Download missing resources as per error messages.  
2. Video stream playback stuttering: Await Gradio's optimization for video streaming.  
3. Model loading issues: Check if weights are downloaded completely.
