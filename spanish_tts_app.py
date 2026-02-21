import os
import json
import shutil
import tempfile
import gradio as gr
import soundfile as sf
import numpy as np

from cached_path import cached_path
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    transcribe,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VOICES_DIR = os.path.join(os.path.dirname(__file__), "voices")
VOICES_JSON = os.path.join(VOICES_DIR, "voices.json")

os.makedirs(VOICES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

print("Cargando vocoder...")
vocoder = load_vocoder()

print("Cargando modelo F5-Spanish...")
ckpt_path = str(cached_path("hf://jpgallegoar/F5-Spanish/model_1250000.safetensors"))
vocab_path = str(cached_path("hf://jpgallegoar/F5-Spanish/vocab.txt"))
model = load_model(DiT, F5TTS_model_cfg, ckpt_path, vocab_file=vocab_path)

print("Modelo listo.")

# ---------------------------------------------------------------------------
# Voice storage helpers
# ---------------------------------------------------------------------------

def load_voices() -> dict:
    if os.path.exists(VOICES_JSON):
        with open(VOICES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_voices(voices: dict):
    with open(VOICES_JSON, "w", encoding="utf-8") as f:
        json.dump(voices, f, ensure_ascii=False, indent=2)


def voice_names() -> list:
    return list(load_voices().keys())

# ---------------------------------------------------------------------------
# Tab 1 — Generar Audio
# ---------------------------------------------------------------------------

def on_voice_select(voice_name):
    """Auto-populate ref audio path and transcript when a saved voice is chosen."""
    if not voice_name:
        return None, ""
    voices = load_voices()
    if voice_name not in voices:
        return None, ""
    entry = voices[voice_name]
    audio_path = os.path.join(os.path.dirname(__file__), entry["audio"])
    return audio_path, entry["transcript"]


def generate_audio(ref_audio, ref_transcript, gen_text, speed):
    if not ref_audio:
        raise gr.Error("Por favor sube o selecciona un audio de referencia.")
    if not gen_text.strip():
        raise gr.Error("Por favor ingresa el texto a generar.")

    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(
        ref_audio, ref_transcript, language="es"
    )

    audio_out, sample_rate, _ = infer_process(
        ref_audio_proc,
        ref_text_proc,
        gen_text,
        model,
        vocoder,
        speed=speed,
    )

    # Return as numpy array with sample rate so Gradio can play it
    return sample_rate, audio_out

# ---------------------------------------------------------------------------
# Tab 2 — Gestionar Voces
# ---------------------------------------------------------------------------

def voices_table_md():
    voices = load_voices()
    if not voices:
        return "_No hay voces guardadas._"
    lines = ["| Nombre | Transcripción |", "|--------|---------------|"]
    for name, entry in voices.items():
        transcript = entry.get("transcript", "")[:80]
        lines.append(f"| {name} | {transcript} |")
    return "\n".join(lines)


def save_voice(name, audio_path, transcript):
    name = name.strip()
    if not name:
        return "Error: ingresa un nombre para la voz.", gr.update(), gr.update()
    if not audio_path:
        return "Error: sube un audio de referencia.", gr.update(), gr.update()

    # Auto-transcribe if no transcript provided
    if not transcript.strip():
        transcript = transcribe(audio_path, language="es")

    # Copy audio into voices dir
    ext = os.path.splitext(audio_path)[1] or ".wav"
    dest_filename = f"{name}{ext}"
    dest_path = os.path.join(VOICES_DIR, dest_filename)
    shutil.copy2(audio_path, dest_path)

    voices = load_voices()
    voices[name] = {
        "audio": os.path.join("voices", dest_filename),
        "transcript": transcript,
    }
    save_voices(voices)

    names = voice_names()
    status = f"Voz '{name}' guardada correctamente."
    return status, gr.update(choices=names, value=None), gr.update(choices=names, value=None)


def delete_voice(name):
    if not name:
        return "Selecciona una voz para eliminar.", gr.update(), gr.update()
    voices = load_voices()
    if name not in voices:
        return f"Voz '{name}' no encontrada.", gr.update(), gr.update()

    # Remove audio file
    audio_rel = voices[name].get("audio", "")
    audio_abs = os.path.join(os.path.dirname(__file__), audio_rel)
    if os.path.exists(audio_abs):
        os.remove(audio_abs)

    del voices[name]
    save_voices(voices)

    names = voice_names()
    status = f"Voz '{name}' eliminada."
    return status, gr.update(choices=names, value=None), gr.update(choices=names, value=None)

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="F5-TTS Español") as app:
    gr.Markdown("# F5-TTS Español")

    with gr.Tabs():
        # ── Tab 1 ──────────────────────────────────────────────────────────
        with gr.Tab("Generar Audio"):
            with gr.Row():
                voice_dropdown = gr.Dropdown(
                    label="Seleccionar voz guardada",
                    choices=voice_names(),
                    value=None,
                    interactive=True,
                )

            ref_audio_input = gr.Audio(
                label="Audio de referencia",
                type="filepath",
            )
            ref_transcript_input = gr.Textbox(
                label="Transcripción del audio de referencia",
                placeholder="Se rellena automáticamente al seleccionar una voz guardada.",
                lines=2,
            )
            gen_text_input = gr.Textbox(
                label="Texto a generar",
                placeholder="Escribe el texto en español que quieres sintetizar...",
                lines=4,
            )
            speed_slider = gr.Slider(
                label="Velocidad",
                minimum=0.3,
                maximum=2.0,
                value=1.0,
                step=0.05,
            )
            generate_btn = gr.Button("Generar", variant="primary")
            audio_output = gr.Audio(label="Audio generado")

            # Wire voice selection → auto-populate fields
            voice_dropdown.change(
                fn=on_voice_select,
                inputs=[voice_dropdown],
                outputs=[ref_audio_input, ref_transcript_input],
            )

            generate_btn.click(
                fn=generate_audio,
                inputs=[ref_audio_input, ref_transcript_input, gen_text_input, speed_slider],
                outputs=[audio_output],
            )

        # ── Tab 2 ──────────────────────────────────────────────────────────
        with gr.Tab("Gestionar Voces"):
            voices_md = gr.Markdown(voices_table_md())

            gr.Markdown("### Guardar nueva voz")
            new_voice_name = gr.Textbox(label="Nombre de la voz", placeholder="Ej: Maria")
            new_voice_audio = gr.Audio(label="Audio de referencia", type="filepath")
            new_voice_transcript = gr.Textbox(
                label="Transcripción (opcional — se auto-transcribe en español si se deja vacío)",
                lines=2,
            )
            save_btn = gr.Button("Guardar Voz", variant="primary")

            gr.Markdown("### Eliminar voz")
            delete_dropdown = gr.Dropdown(
                label="Seleccionar voz a eliminar",
                choices=voice_names(),
                value=None,
                interactive=True,
            )
            delete_btn = gr.Button("Eliminar Voz", variant="stop")

            status_box = gr.Textbox(label="Estado", interactive=False)

            def save_and_refresh(name, audio, transcript):
                status, dd1, dd2 = save_voice(name, audio, transcript)
                table = voices_table_md()
                return status, table, dd1, dd2

            def delete_and_refresh(name):
                status, dd1, dd2 = delete_voice(name)
                table = voices_table_md()
                return status, table, dd1, dd2

            save_btn.click(
                fn=save_and_refresh,
                inputs=[new_voice_name, new_voice_audio, new_voice_transcript],
                outputs=[status_box, voices_md, voice_dropdown, delete_dropdown],
            )

            delete_btn.click(
                fn=delete_and_refresh,
                inputs=[delete_dropdown],
                outputs=[status_box, voices_md, voice_dropdown, delete_dropdown],
            )

if __name__ == "__main__":
    app.launch(share=False)
