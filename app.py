import io
import wave
import numpy as np
import streamlit as st
from pedalboard import (Pedalboard, Reverb, Delay, LowShelfFilter, HighShelfFilter,
                        LowpassFilter, HighpassFilter, Compressor, Limiter, time_stretch)
from pedalboard.io import AudioFile

st.set_page_config(page_title="Slowed + Reverb Studio", layout="wide")
st.title("Slowed + Reverb Studio")

# ---------- Reverb presets: (room_size, damping, wet, dry) ----------
PRESETS = {
    "Small room":     (0.30, 0.50, 0.20, 0.90),
    "Hall":           (0.70, 0.40, 0.30, 0.80),
    "Cathedral":      (0.95, 0.30, 0.40, 0.70),
    "Dreamy cloud":   (0.90, 0.60, 0.50, 0.60),
    "Infinite space": (1.00, 0.20, 0.55, 0.55),
    "Custom":         (0.70, 0.50, 0.30, 0.80),
}

file = st.file_uploader("Upload a song", type=["mp3", "wav", "flac", "ogg"])

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Slow")
    speed = st.slider("Speed (1.0 = original)", 0.50, 1.00, 0.85, 0.01)
    mode = st.radio("Slow style", ["Classic (slower + lower pitch)", "Keep original pitch"])

with c2:
    st.subheader("Reverb")
    preset = st.selectbox("Reverb type", list(PRESETS.keys()), index=3)
    p = PRESETS[preset]
    manual = preset == "Custom"
    room = st.slider("Room size", 0.0, 1.0, p[0], 0.01, disabled=not manual)
    damp = st.slider("Damping (higher = darker tail)", 0.0, 1.0, p[1], 0.01, disabled=not manual)
    wet = st.slider("Reverb amount (wet)", 0.0, 1.0, p[2], 0.01, disabled=not manual)
    dry = st.slider("Original signal (dry)", 0.0, 1.0, p[3], 0.01, disabled=not manual)
    if not manual:
        room, damp, wet, dry = p[0], p[1], p[2], p[3]

with c3:
    st.subheader("Soundstage")
    width = st.slider("Stereo width (1.0 = normal)", 0.0, 2.5, 1.4, 0.05)
    pan_on = st.checkbox("8D auto-pan (sound circles your head)")
    pan_speed = st.slider("Pan cycle length (seconds)", 2.0, 20.0, 8.0, 0.5, disabled=not pan_on)
    pan_depth = st.slider("Pan depth", 0.0, 1.0, 0.6, 0.05, disabled=not pan_on)

st.subheader("Tone and polish")
t1, t2, t3, t4 = st.columns(4)
with t1:
    bass = st.slider("Bass boost (dB)", -6.0, 12.0, 3.0, 0.5)
with t2:
    treble = st.slider("Treble (dB)", -12.0, 6.0, -2.0, 0.5)
with t3:
    lowpass = st.slider("Muffle (low-pass Hz, 20000 = off)", 2000, 20000, 20000, 500)
with t4:
    glue = st.checkbox("Compressor (glue)", value=True)

e1, e2, e3 = st.columns(3)
with e1:
    echo_on = st.checkbox("Add echo")
with e2:
    echo_time = st.slider("Echo time (s)", 0.05, 1.0, 0.35, 0.01, disabled=not echo_on)
with e3:
    echo_mix = st.slider("Echo amount", 0.0, 0.6, 0.2, 0.01, disabled=not echo_on)


def apply_width(audio, w):
    mid = (audio[0] + audio[1]) / 2
    side = (audio[0] - audio[1]) / 2 * w
    return np.stack([mid + side, mid - side])


def apply_pan(audio, sr, cycle, depth):
    t = np.arange(audio.shape[1]) / sr
    pos = np.sin(2 * np.pi * t / cycle) * depth
    left = np.sqrt((1 - pos) / 2)
    right = np.sqrt((1 + pos) / 2)
    mono_gain = np.sqrt(2) / 2
    return np.stack([audio[0] * left / mono_gain, audio[1] * right / mono_gain])


def to_wav_bytes(audio, sample_rate):
    """audio shape: (2, samples), floats between -1 and 1"""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio.T * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


if file and st.button("Make it", type="primary"):
    with st.spinner("Processing..."):
        with AudioFile(io.BytesIO(file.read())) as f:
            audio = f.read(f.frames).astype(np.float32)
            sr = int(f.samplerate)

        if audio.shape[0] == 1:
            audio = np.vstack([audio, audio])
        audio = audio[:2]

        if mode.startswith("Keep"):
            audio = time_stretch(audio, sr, stretch_factor=speed)
            out_sr = sr
        else:
            out_sr = int(sr * speed)

        audio = np.pad(audio, ((0, 0), (0, sr * 4)))  # room for reverb tail

        fx = [HighpassFilter(25)]
        if bass != 0:
            fx.append(LowShelfFilter(cutoff_frequency_hz=150, gain_db=bass))
        if treble != 0:
            fx.append(HighShelfFilter(cutoff_frequency_hz=6000, gain_db=treble))
        if lowpass < 20000:
            fx.append(LowpassFilter(lowpass))
        if glue:
            fx.append(Compressor(threshold_db=-18, ratio=2.0))
        if echo_on:
            fx.append(Delay(delay_seconds=echo_time, feedback=0.3, mix=echo_mix))
        fx.append(Reverb(room_size=room, damping=damp, wet_level=wet, dry_level=dry, width=1.0))
        out = Pedalboard(fx)(audio, sr)

        out = apply_width(out, width)
        if pan_on:
            out = apply_pan(out, sr, pan_speed, pan_depth)

        out = Pedalboard([Limiter(threshold_db=-1.0)])(out.astype(np.float32), sr)

        # keep the result so it survives the page refresh when you click Download
        st.session_state["result"] = to_wav_bytes(out, out_sr)

if "result" in st.session_state:
    st.success("Done!")
    st.audio(st.session_state["result"], format="audio/wav")
    st.download_button("Download WAV", st.session_state["result"],
                       "slowed_reverb.wav", "audio/wav")
