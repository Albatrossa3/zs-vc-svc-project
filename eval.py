# eval.py
import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf

# Reuse loaders from your SVC code
from SVC.svc import load_rmvpe, load_SV_model


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()


def list_pairs(mode_dir: str, prefix: str):
    """
    Read files like:
      - {prefix}-source-{id}.wav
      - {prefix}-ref-{id}.wav
      - {prefix}-{src_id}-{ref_id}.wav
    and return only pairs that have both source/ref present.
    """
    src_pat = re.compile(rf"^{re.escape(prefix)}-source-(\d+)\.wav$", re.I)
    ref_pat = re.compile(rf"^{re.escape(prefix)}-ref-(\d+)\.wav$", re.I)
    res_pat = re.compile(rf"^{re.escape(prefix)}-(\d+)-(\d+)\.wav$", re.I)

    files = os.listdir(mode_dir) if os.path.isdir(mode_dir) else []
    srcs = {int(m.group(1)) for f in files if (m := src_pat.match(f))}
    refs = {int(m.group(1)) for f in files if (m := ref_pat.match(f))}
    pairs = {(int(m.group(1)), int(m.group(2))) for f in files if (m := res_pat.match(f))}
    pairs = {(s, r) for (s, r) in pairs
             if f"{prefix}-source-{s}.wav" in files and f"{prefix}-ref-{r}.wav" in files}
    return sorted(pairs)


def load_campplus(camp_ckpt_path: str):
    """
    Use your existing loader so we get the exact model/weights you use in-app.
    """
    model = load_SV_model(camp_ckpt_path, DEVICE)  # returns CAMPPlus already on device
    model.eval()
    return model


@torch.no_grad()
def campplus_embed(model, wav_path: str):
    """
    CAMPPlus embeddings from 16 kHz fbank.
    """
    wav, _ = librosa.load(wav_path, sr=16000)
    x = torch.tensor(wav).unsqueeze(0).to(DEVICE)  # (1, T)
    fb = torchaudio.compliance.kaldi.fbank(
        x,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000
    )  # (T, 80)
    fb = fb - fb.mean(dim=0, keepdim=True)
    emb = model(fb.unsqueeze(0))  # (1, 192)
    return emb.squeeze(0)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float((a * b).sum())


def duration_sec(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def duration_ratio(conv_path: str, src_path: str) -> float:
    d_c = duration_sec(conv_path)
    d_s = duration_sec(src_path)
    return d_c / (d_s + 1e-9)


def rmvpe_f0(rmvpe, wav_path: str):
    wav, _ = librosa.load(wav_path, sr=16000)
    return rmvpe.infer_from_audio(wav.astype(np.float32))


def f0_keymedian(f0: np.ndarray):
    voiced = f0[f0 > 1]
    if len(voiced) == 0:
        return np.nan
    return np.median(np.log2(voiced))


def f0_metrics_key_norm(f0_conv: np.ndarray, f0_src: np.ndarray):
    """
    Key-normalized F0 error: align medians in log2-domain then compute contour error.
    Returns (median absolute error in semitones, RMSE in cents).
    """
    L = min(len(f0_conv), len(f0_src))
    a = f0_conv[:L]
    b = f0_src[:L]
    m_a = f0_keymedian(a)
    m_b = f0_keymedian(b)
    if np.isnan(m_a) or np.isnan(m_b):
        return np.nan, np.nan
    shift = m_a - m_b  # log2 shift
    b_norm = b * (2 ** shift)
    mask = (a > 1) & (b_norm > 1)
    if mask.sum() < 10:
        return np.nan, np.nan
    semitone_err = 12 * np.log2((a[mask] + 1e-5) / (b_norm[mask] + 1e-5))
    med_abs_semi = float(np.median(np.abs(semitone_err)))
    rmse_cents = float(np.sqrt(np.mean((100 * semitone_err) ** 2)))
    return med_abs_semi, rmse_cents


def f0_key_offset_to_ref(f0_conv: np.ndarray, f0_ref: np.ndarray):
    """
    Optional: report median key offset conv vs ref in semitones.
    """
    m_c = f0_keymedian(f0_conv)
    m_r = f0_keymedian(f0_ref)
    if np.isnan(m_c) or np.isnan(m_r):
        return np.nan
    return float(12 * (m_c - m_r))


def evaluate(conversion_demo_dir: str) -> dict:
    """
    Walk conversion_demo/{vc,svc} and compute:
      VC:  spk_sim (conv vs ref), dur_ratio
      SVC: spk_sim, f0_med_err_semi (conv vs source), f0_rmse_cents (conv vs source),
           key_offset_vs_ref_semi (conv vs ref), dur_ratio
    """
    vc_dir = os.path.join(conversion_demo_dir, "vc")
    svc_dir = os.path.join(conversion_demo_dir, "svc")

    # Tools
    camp_ckpt = os.path.join("SVC", "campplus_cn_common.bin")
    rmvpe_path = os.path.join("SVC", "models", "F0", "rmvpe.pt")
    camp = load_campplus(camp_ckpt)
    rmvpe = load_rmvpe(rmvpe_path, DEVICE)

    results = []

    # VC
    if os.path.isdir(vc_dir):
        for (s, r) in list_pairs(vc_dir, "vc"):
            src = os.path.join(vc_dir, f"vc-source-{s}.wav")
            ref = os.path.join(vc_dir, f"vc-ref-{r}.wav")
            con = os.path.join(vc_dir, f"vc-{s}-{r}.wav")
            # Speaker similarity (converted vs reference)
            emb_c = campplus_embed(camp, con)
            emb_r = campplus_embed(camp, ref)
            spk_sim = cosine_sim(emb_c, emb_r)
            # Duration ratio (converted vs source)
            dr = duration_ratio(con, src)

            results.append({
                "mode": "VC",
                "pair": f"{s}-{r}",
                "spk_sim": spk_sim,
                "dur_ratio": dr,
                "source": src,
                "reference": ref,
                "converted": con
            })

    # SVC
    if os.path.isdir(svc_dir):
        for (s, r) in list_pairs(svc_dir, "svc"):
            src = os.path.join(svc_dir, f"svc-source-{s}.wav")
            ref = os.path.join(svc_dir, f"svc-ref-{r}.wav")
            con = os.path.join(svc_dir, f"svc-{s}-{r}.wav")
            # Speaker similarity (converted vs reference)
            emb_c = campplus_embed(camp, con)
            emb_r = campplus_embed(camp, ref)
            spk_sim = cosine_sim(emb_c, emb_r)
            # F0 metrics (conv vs source, key-normalized)
            f0_conv = rmvpe_f0(rmvpe, con)
            f0_src = rmvpe_f0(rmvpe, src)
            f0_ref = rmvpe_f0(rmvpe, ref)
            f0_med_err_semi, f0_rmse_cents = f0_metrics_key_norm(f0_conv, f0_src)
            key_offset_vs_ref_semi = f0_key_offset_to_ref(f0_conv, f0_ref)
            # Duration ratio
            dr = duration_ratio(con, src)

            results.append({
                "mode": "SVC",
                "pair": f"{s}-{r}",
                "spk_sim": spk_sim,
                "f0_med_err_semi": f0_med_err_semi,
                "f0_rmse_cents": f0_rmse_cents,
                "key_offset_vs_ref_semi": key_offset_vs_ref_semi,
                "dur_ratio": dr,
                "source": src,
                "reference": ref,
                "converted": con
            })

    out = {
        "results": results,
        "notes": (
            "VC rows: speaker similarity (conv vs ref), duration ratio only. "
            "SVC rows: add F0 contour error (conv vs source, key-normalized) and key offset vs ref. "
            "Higher spk_sim is better. Lower F0 errors are better. dur_ratio ~ 1.0 indicates timing preserved. "
            "WER/ASR omitted due to instability on singing and multilingual inputs."
        )
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-dir", type=str, default="conversion_demo",
                        help="Path to conversion_demo folder")
    parser.add_argument("--out", type=str, default="eval_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    with torch.no_grad():
        out = evaluate(args.demo_dir)

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()