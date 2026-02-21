module.exports = {
  requires: {
    bundle: "ai",
  },
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          "PYTORCH_ENABLE_MPS_FALLBACK": 1,
          "HF_HUB_ENABLE_HF_TRANSFER": 0
        },
        path: "app",
        message: [
          "python ../spanish_tts_app.py"
        ],
        on: [{
          "event": "/http:\/\/\\S+/",
          "done": true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"
      }
    }
  ]
}
