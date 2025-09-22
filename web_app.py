# web_app.py
import os
import yaml
import torch
import librosa
import torchaudio
import numpy as np
import gradio as gr
import gradio as gr
import soundfile as sf
import subprocess
import time
import hashlib
from pathlib import Path

# set the cache directory for the inputs
TMP_DIR = os.path.join(os.getcwd(), "cache_inputs")
os.makedirs(TMP_DIR, exist_ok=True)

# set the hash name
def _hash_name(path: str, suffix: str) -> str:
    base = Path(path).stem
    h = hashlib.md5((path + str(time.time())).encode()).hexdigest()[:8]
    return f"{base}_{h}{suffix}"

# ensure the input is a wav file
def ensure_wav(file_path: str, target_sr: int | None = None) -> str:
    """
    Convert any input file to WAV on disk for downstream processing.
    If target_sr is provided, resample; else keep original sr.
    """
    y, sr = librosa.load(file_path, sr=target_sr)
    out_name = _hash_name(file_path, ".wav")
    out_path = os.path.join(TMP_DIR, out_name)
    sf.write(out_path, y, sr)
    return out_path

# extract the vocals from the input file
def extract_vocals_demucs(file_path: str) -> str:
    """
    Use Demucs CLI to extract vocals stem.
    Returns path to vocals WAV if successful; otherwise returns a WAV copy of input.
    """
    try:
        out_root = os.path.join(TMP_DIR, "demucs_out")
        os.makedirs(out_root, exist_ok=True)
        # use CLI to avoid tight coupling to internal API surface.
        cmd = [
            "python", "-m", "demucs",
            "--two-stems=vocals",
            file_path,
            "-o", out_root
        ]
        # run the command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        base = Path(file_path).stem
        cand = os.path.join(out_root, "htdemucs", base, "vocals.wav")
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    # fallback: just ensure WAV
    return ensure_wav(file_path)

# reuse loaders/utilities from VC and SVC
from VC.vc import (
    load_vc_model as load_vc_model_base,
    load_SV_model as load_SV_model_base,
    load_vocoder as load_vocoder_base,
    load_token_model as load_token_model_base,
    get_mel_fn as get_mel_fn_base,
    gen_token_featues as gen_token_features_base,
)
from SVC.svc import (
    load_vc_model as load_vc_model_f0,
    load_SV_model as load_SV_model_f0,
    load_vocoder as load_vocoder_f0,
    load_token_model as load_token_model_f0,
    load_rmvpe as load_rmvpe_f0,
    transpose_f0 as transpose_f0,
    get_mel_fn as get_mel_fn_f0,
    gen_token_featues as gen_token_features_f0,
)

# crossfade the two chunks
def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2

# pick the device
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# VC service
class VCService:
    def __init__(self):
        self.device = pick_device()
        # VC (VC, 22k)
        self.vc_cfg_path = os.path.join("VC", "models", "vc", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
        self.vc_ckpt_path = os.path.join("VC", "models", "vc", "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth")
        self.vocoder_path = os.path.join("VC", "models", "vocoder")
        self.tokenizer_path = os.path.join("VC", "models", "tokenizer")
        self.campplus_ckpt = os.path.join("VC", "campplus_cn_common.bin")
        self._loaded = False

    # load the model
    def load(self):
        if self._loaded:
            return
        config = yaml.safe_load(open(self.vc_cfg_path, "r"))
        self.model = load_vc_model_base(self.vc_ckpt_path, config, self.device)
        self.campplus = load_SV_model_base(self.campplus_ckpt, self.device)
        self.vocoder = load_vocoder_base(self.vocoder_path, self.device)
        self.whisper_model, self.whisper_fe = load_token_model_base(self.tokenizer_path, self.device)
        self.to_mel = get_mel_fn_base(config)
        self.hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
        self.sr = config["preprocess_params"]["sr"]
        self.max_context_window = self.sr // self.hop_length * 30
        self.overlap_frame_len = 16
        self.overlap_wave_len = self.overlap_frame_len * self.hop_length
        self.fp16 = True if self.device.type == "cuda" else False
        self._loaded = True

    # convert the audio
    @torch.no_grad()
    @torch.inference_mode()
    def convert(self, source_path, ref_path, diffusion_steps=30, length_adjust=1.0, inference_cfg_rate=0.7):
        self.load()
        # Load audio at model SR (22k)
        src, _ = librosa.load(source_path, sr=self.sr)
        ref, _ = librosa.load(ref_path, sr=self.sr)
        source_audio = torch.tensor(src).unsqueeze(0).float().to(self.device)
        ref_audio = torch.tensor(ref[: self.sr * 25]).unsqueeze(0).float().to(self.device)

        # 16k for Whisper + SV embeddings
        ref_16k = torchaudio.functional.resample(ref_audio, self.sr, 16000)
        src_16k = torchaudio.functional.resample(source_audio, self.sr, 16000)

        S_alt = gen_token_features_base(src_16k, self.whisper_model, self.whisper_fe, self.device)
        S_ori = gen_token_features_base(ref_16k, self.whisper_model, self.whisper_fe, self.device)

        feat_sv = torchaudio.compliance.kaldi.fbank(ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat_sv = feat_sv - feat_sv.mean(dim=0, keepdim=True)
        style = self.campplus(feat_sv.unsqueeze(0))

        src_mel = self.to_mel(source_audio.to(self.device).float())
        ref_mel = self.to_mel(ref_audio.to(self.device).float())
        tgt_len = torch.LongTensor([int(src_mel.size(2) * length_adjust)]).to(src_mel.device)
        ref_len = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

        # No F0 conditioning for VC
        cond, _, _, _, _ = self.model.length_regulator(S_alt, ylens=tgt_len, n_quantizers=3, f0=None)
        prompt, _, _, _, _ = self.model.length_regulator(S_ori, ylens=ref_len, n_quantizers=3, f0=None)

        max_source_window = self.max_context_window - ref_mel.size(2)
        processed_frames = 0
        chunks = []
        previous_chunk = None

        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
            is_last = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt, chunk_cond], dim=1)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.fp16 else torch.float32):
                vc_target = self.model.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(ref_mel.device),
                    ref_mel, style, None, diffusion_steps, inference_cfg_rate=inference_cfg_rate
                )
                vc_target = vc_target[:, :, ref_mel.size(-1):]
                vc_wave = self.vocoder(vc_target)[0]
            # if the wave is a single dimension, convert it to a two dimension
            if vc_wave.ndim == 1:
                vc_wave = vc_wave.unsqueeze(0)
            if processed_frames == 0:
                # if the processed frames is 0, then add the wave to the chunks
                if is_last:
                    chunks.append(vc_wave[0].cpu().numpy())
                    break
                # if the processed frames is not 0, then add the wave to the chunks
                out = vc_wave[0, :-self.overlap_wave_len].cpu().numpy()
                chunks.append(out)
                previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                processed_frames += vc_target.size(2) - self.overlap_frame_len
            elif is_last:
                # if the processed frames is the last, then add the wave to the chunks
                out = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), self.overlap_wave_len)
                chunks.append(out)
                processed_frames += vc_target.size(2) - self.overlap_frame_len
                break
            else:
                # if not the last, then add the wave to the chunks
                out = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-self.overlap_wave_len].cpu().numpy(), self.overlap_wave_len)
                chunks.append(out)
                previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                processed_frames += vc_target.size(2) - self.overlap_frame_len

        audio = np.concatenate(chunks).astype(np.float32)
        return (self.sr, audio)

class SVCService:
    def __init__(self):
        self.device = pick_device()
        # SVC (SVC, 44k, F0)
        self.vc_cfg_path = os.path.join("SVC", "models", "vc", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
        self.vc_ckpt_path = os.path.join("SVC", "models", "vc", "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth")
        self.vocoder_path = os.path.join("SVC", "models", "vocoder")
        self.tokenizer_path = os.path.join("SVC", "models", "tokenizer")
        self.campplus_ckpt = os.path.join("SVC", "campplus_cn_common.bin")
        self.rmvpe_path = os.path.join("SVC", "models", "F0", "rmvpe.pt")
        self._loaded = False

    # load the model
    def load(self):
        if self._loaded:
            return
        config = yaml.safe_load(open(self.vc_cfg_path, "r"))
        self.model = load_vc_model_f0(self.vc_ckpt_path, config, self.device)
        self.campplus = load_SV_model_f0(self.campplus_ckpt, self.device)
        self.vocoder = load_vocoder_f0(self.vocoder_path, self.device)
        self.whisper_model, self.whisper_fe = load_token_model_f0(self.tokenizer_path, self.device)
        self.rmvpe = load_rmvpe_f0(self.rmvpe_path, self.device)
        self.to_mel = get_mel_fn_f0(config)
        self.hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
        self.sr = config["preprocess_params"]["sr"]
        self.max_context_window = self.sr // self.hop_length * 30
        self.overlap_frame_len = 16
        self.overlap_wave_len = self.overlap_frame_len * self.hop_length
        self.fp16 = True if self.device.type == "cuda" else False
        self._loaded = True

    @torch.no_grad()
    @torch.inference_mode()
    def convert(self, source_path, ref_path, diffusion_steps=30, length_adjust=1.0, inference_cfg_rate=0.7, pitch_shift=0):
        self.load()
        # load audio at model SR (44k)
        src, _ = librosa.load(source_path, sr=self.sr)
        ref, _ = librosa.load(ref_path, sr=self.sr)
        source_audio = torch.tensor(src).unsqueeze(0).float().to(self.device)
        ref_audio = torch.tensor(ref[: self.sr * 25]).unsqueeze(0).float().to(self.device)

        # 16k for Whisper + SV embeddings + F0
        ref_16k = torchaudio.functional.resample(ref_audio, self.sr, 16000)
        src_16k = torchaudio.functional.resample(source_audio, self.sr, 16000)

        S_alt = gen_token_features_f0(src_16k, self.whisper_model, self.whisper_fe, self.device)
        S_ori = gen_token_features_f0(ref_16k, self.whisper_model, self.whisper_fe, self.device)

        feat_sv = torchaudio.compliance.kaldi.fbank(ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat_sv = feat_sv - feat_sv.mean(dim=0, keepdim=True)
        style = self.campplus(feat_sv.unsqueeze(0))

        src_mel = self.to_mel(source_audio.to(self.device).float())
        ref_mel = self.to_mel(ref_audio.to(self.device).float())
        tgt_len = torch.LongTensor([int(src_mel.size(2) * length_adjust)]).to(src_mel.device)
        ref_len = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

        # F0 extraction in Hz, with automatic key alignment (raw Hz passed to length_regulator)
        f0_ref_hz = self.rmvpe.infer_from_audio(ref_16k.squeeze(0).cpu().numpy())
        f0_src_hz = self.rmvpe.infer_from_audio(src_16k.squeeze(0).cpu().numpy())
        ref_med = np.median(f0_ref_hz[f0_ref_hz > 0])
        src_med = np.median(f0_src_hz[f0_src_hz > 0])
        auto_shift = 12 * np.log2(max(ref_med, 1e-5) / max(src_med, 1e-5))
        total_shift = auto_shift + float(pitch_shift)
        f0_src_hz_shifted = transpose_f0(f0_src_hz, total_shift)
        F0_ori = torch.from_numpy(f0_ref_hz).unsqueeze(0).to(self.device)
        F0_alt = torch.from_numpy(f0_src_hz_shifted).unsqueeze(0).to(self.device)

        cond, _, _, _, _ = self.model.length_regulator(S_alt, ylens=tgt_len, n_quantizers=3, f0=F0_alt)
        prompt, _, _, _, _ = self.model.length_regulator(S_ori, ylens=ref_len, n_quantizers=3, f0=F0_ori)

        max_source_window = self.max_context_window - ref_mel.size(2)
        processed_frames = 0
        chunks = []
        previous_chunk = None

        # generate chunk by chunk and stream the output
        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
            is_last = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt, chunk_cond], dim=1)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.fp16 else torch.float32):
                vc_target = self.model.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(ref_mel.device),
                    ref_mel, style, None, diffusion_steps, inference_cfg_rate=inference_cfg_rate
                )
                vc_target = vc_target[:, :, ref_mel.size(-1):]
                vc_wave = self.vocoder(vc_target)[0]

            if vc_wave.ndim == 1:
                vc_wave = vc_wave.unsqueeze(0)
            if processed_frames == 0:
                if is_last:
                    chunks.append(vc_wave[0].cpu().numpy())
                    break
                out = vc_wave[0, :-self.overlap_wave_len].cpu().numpy()
                chunks.append(out)
                previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                processed_frames += vc_target.size(2) - self.overlap_frame_len
            elif is_last:
                out = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), self.overlap_wave_len)
                chunks.append(out)
                processed_frames += vc_target.size(2) - self.overlap_frame_len
                break
            else:
                out = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-self.overlap_wave_len].cpu().numpy(), self.overlap_wave_len)
                chunks.append(out)
                previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                processed_frames += vc_target.size(2) - self.overlap_frame_len

        audio = np.concatenate(chunks).astype(np.float32)
        return (self.sr, audio)

# single, shared app with a mode switch
vc_service = VCService()
svc_service = SVCService()

def do_convert(mode, source_file, reference_file, diffusion_steps, length_adjust, cfg_rate,
               pitch_shift, extract_vocals, extract_scope):
    if not source_file or not reference_file:
        return None

    # prepare inputs: vocal separation and ensure WAV
    src_path, ref_path = source_file, reference_file
    if extract_vocals:
        if extract_scope in ("Both", "Source only"):
            src_path = extract_vocals_demucs(src_path)
        else:
            src_path = ensure_wav(src_path)
        if extract_scope in ("Both", "Reference only"):
            ref_path = extract_vocals_demucs(ref_path)
        else:
            ref_path = ensure_wav(ref_path)
    else:
        src_path = ensure_wav(src_path)
        ref_path = ensure_wav(ref_path)

    if mode == "VC":
        return vc_service.convert(src_path, ref_path, diffusion_steps, length_adjust, cfg_rate)
    else:
        return svc_service.convert(src_path, ref_path, diffusion_steps, length_adjust, cfg_rate, pitch_shift)


def main():
    with gr.Blocks(title="VC/SVC Demo") as demo:
        gr.Markdown(
            "**Zero-shot VC/SVC Demo**  \n"
            "- Convert audio to the reference voice without training.  \n"
            "- Choose VC (voice conversion) or SVC (Singing voice conversion).  \n"
            "- Upload a source and a reference audio.  \n"
            "- Optional: Extract vocals to improve conversion.  \n"
            "Note: Vocal extraction increases processing time on first run."
        )
        with gr.Row():
            mode = gr.Radio(choices=["VC", "SVC"], value="SVC", label="Mode")
        with gr.Row():
            src = gr.Audio(type="filepath", label="Source audio")
            ref = gr.Audio(type="filepath", label="Reference audio")
        with gr.Row():
            steps = gr.Slider(30, 150, value=70, step=1, label="Diffusion steps",
                              info="Higher → better quality but slower. 50–100 is typical best quality.")
            pitch_shift = gr.Slider(-24, 24, value=0, step=1, label="Semitone shift (SVC only)",
                                    info="Transpose pitch for SVC on top of auto key alignment (−24..+24).")
            length_adj = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Length adjust",
                                   info="<1.0 speeds up, >1.0 slows down the source timing.")
            cfg = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Inference CFG rate",
                            info="Controls variation/diversity; 0.6–0.8 usually works well.")
        with gr.Row():
            extract_vocals = gr.Checkbox(value=False, label="Extract vocals",
                                         info="Separate vocals from mixture before conversion; converts to WAV automatically.")
            extract_scope = gr.Dropdown(choices=["Both", "Source only", "Reference only"],
                                        value="Both", label="Apply extraction to")
        btn = gr.Button("Convert")
        out = gr.Audio(label="Converted audio", type="numpy")
        btn.click(
            fn=do_convert,
            inputs=[mode, src, ref, steps, length_adj, cfg, pitch_shift, extract_vocals, extract_scope],
            outputs=[out]
        )
    demo.launch()

if __name__ == "__main__":
    main()