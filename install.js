module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/SWivid/F5-TTS app",
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
          // xformers: true
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -e .",
          "uv pip uninstall torchcodec",
          "uv pip install hf_xet"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python -c \"content = open('src/f5_tts/infer/infer_gradio.py').read(); content = content.replace('hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors', 'hf://jpgallegoar/F5-Spanish/model_1250000.safetensors').replace('hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt', 'hf://jpgallegoar/F5-Spanish/vocab.txt'); open('src/f5_tts/infer/infer_gradio.py', 'w').write(content); print('Patched default model to Spanish F5-TTS')\"",
          "python -c \"content = open('src/f5_tts/infer/utils_infer.py').read(); content = content.replace('ref_text = transcribe(ref_audio)', 'ref_text = transcribe(ref_audio, language=\\'es\\')'); open('src/f5_tts/infer/utils_infer.py', 'w').write(content); print('Patched STT to use Spanish language')\""
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "Click the 'start' tab to get started!"
      }
    }
  ]
}
