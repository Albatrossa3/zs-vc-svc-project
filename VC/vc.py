# vc.py
import yaml
import os
from modules.commons import build_model, load_checkpoint, recursive_munch, str2bool
import torch
import librosa
import torchaudio
import numpy as np
import soundfile as sf
from modules.campplus.DTDNN import CAMPPlus
from modules.bigvgan import bigvgan
from modules.bigvgan import bigvgan
from transformers import AutoFeatureExtractor, WhisperModel


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


# Loading the VC model
def load_vc_model(file_model,config,device):
    
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    
     # Load checkpoints
    dit_checkpoint_path = file_model
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    return model


# Voiceprint model
def load_SV_model(file_model,device):
    
    campplus_ckpt_path = file_model
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)
    return campplus_model


# Loading the vocoder
def load_vocoder(path_model,device):
    bigvgan_name = path_model
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
    # remove weight norm in the model and set to eval mode
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval().to(device)
    return bigvgan_model


# Text feature extractor
def load_token_model(path_model,device):
    whisper_name = path_model
    whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
    del whisper_model.decoder
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)
    return whisper_model,whisper_feature_extractor


def semantic_fn(waves_16k,whisper_model,whisper_feature_extractor,device):

    ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()], # get whisper model input features
                                            return_tensors="pt",
                                            return_attention_mask=True)
    # The whisper model inputs data for 30 seconds each time, so a mask is required when it's too short.
    ori_input_features = whisper_model._mask_input_features(ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
    with torch.no_grad():
        ori_outputs = whisper_model.encoder(
            ori_input_features.to(whisper_model.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    S_ori = ori_outputs.last_hidden_state.to(torch.float32)
    # Final output: 320 sample points, corresponding to 1 frame
    # Input mel feature hop_size 160, encoder has a CNN, stride-size =2
    S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1] 
    return S_ori


# Acoustic feature extraction
def get_mel_fn(config):
    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": config["preprocess_params"]["sr"],
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
    return to_mel


def gen_token_featues(wavs_16k,whisper_model,whisper_feature_extractor,device):
    if wavs_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(wavs_16k,whisper_model,whisper_feature_extractor,device)
       
    else:
        # speech is too long so it is processed in segments with 5 seconds overlap
        overlapping_time = 5  
        S_alt_list = []
        
        buffer = None
        traversed_time = 0
        while traversed_time < wavs_16k.size(-1):
            if buffer is None:  # first chunk
                chunk = wavs_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat([buffer, wavs_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]], dim=-1)
            S_alt = semantic_fn(chunk,whisper_model,whisper_feature_extractor,device)
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time:]) # 1s corresponds to 50 frames of features
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    return S_alt


# Load various models in sequence
if __name__ == "__main__":
    device = torch.device('cuda:0') # define device
    path_vc_model = 'models/vc'
    # read config parameters
    dit_config_path =os.path.join(path_vc_model,'config_dit_mel_seed_uvit_whisper_small_wavenet.yml')
    config = yaml.safe_load(open(dit_config_path, "r"))
    
    # read voice conversion model
    dit_checkpoint_path = os.path.join(path_vc_model,'DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth')
    model= load_vc_model(dit_checkpoint_path,config,device)

    # read SV model
    campplus_ckpt_path = 'campplus_cn_common.bin'
    campplus_model = load_SV_model(campplus_ckpt_path,device)

    # read vocoder
    path_vocoder = 'models/vocoder'
    vocoder_fn = load_vocoder(path_vocoder,device)

    # read token model
    whisper_name = 'models/tokenizer'
    token_model,token_extractor =load_token_model(whisper_name,device)

    # get the preprocessing mel feature extraction function
    to_mel = get_mel_fn(config)

    # Define some global parameters in the VC process
    length_adjust = 1  # Adjust the speech speed of the conversion results
    diffusion_steps =100 # diffusion steps
    # The inference_cfg rate is refering to the diffusion process, where there is a completely empty data input into the DIT, and the result is the weighted sum of the two partial outputs, so as to increase the diversity of the output results.
    inference_cfg_rate = 0.7 
    fp16 = True # inference with GPU / False for CPU
    
    # When the source speech is too long, some parameters are required for processing
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"] # 22k
    overlap_frame_len = 16 # When inferring, the source speech is too long and the overlapping part between the two segments needs to be inferred segment by segment
    max_context_window = sr // hop_length * 30  #  ref+src 30 seconds max
    overlap_wave_len = overlap_frame_len * hop_length 

    # Below are conversion steps

    # Read the source and target voices, and set the sampling rate to sr
    source = "examples/source/s2.wav"
    target = "examples/reference/trump.wav"
    source_audio,fs = librosa.load(source, sr=sr) # Read the source and target voices and resample them to sr, 22K
    ref_audio,fs = librosa.load(target, sr=sr)

    with torch.no_grad(),torch.inference_mode():
        inference_module = model
        mel_fn = to_mel

        # Convert to tensor format
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
        ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)  # Reference audio 25 seconds max

        # Resample to 16k for token feature extraction
        ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000) 
        src_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

        # Extract token features
        S_alt = gen_token_featues(src_waves_16k,token_model,token_extractor,device)
        S_ori = gen_token_featues(ref_waves_16k,token_model,token_extractor,device)

        # Extract SV features from ref
        feat_SV = torchaudio.compliance.kaldi.fbank(ref_waves_16k,
                                                   num_mel_bins=80,
                                                   dither=0,
                                                   sample_frequency=16000)
        feat_SV = feat_SV - feat_SV.mean(dim=0, keepdim=True)
        style = campplus_model(feat_SV.unsqueeze(0))

        # Extract mel features for VC from the original source and ref
        src_mel = mel_fn(source_audio.to(device).float())
        ref_mel = mel_fn(ref_audio.to(device).float())

        src_target_lengths = torch.LongTensor([int(src_mel.size(2) * length_adjust)]).to(src_mel.device)
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

        # draft2. skip f0 adjustments for vc
        F0_ori = None
        F0_alt = None
        shifted_f0_alt = None

        # Adjust the length to make the length of token feature and mel consistent
        cond, _, _, _, _ = inference_module.length_regulator(S_alt, ylens=src_target_lengths, n_quantizers=3, f0=shifted_f0_alt)
        prompt_condition, _, _, _, _ = inference_module.length_regulator(S_ori, ylens=ref_target_lengths, n_quantizers=3, f0=F0_ori)

        # maximum length of the source speech that can be converted
        max_source_window = max_context_window - ref_mel.size(2)

        processed_frames = 0
        generated_wave_chunks = []
        # generate chunk by chunk and stream the output
        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
                # Voice Conversion
                vc_target = inference_module.cfm.inference(cat_condition,
                                                        torch.LongTensor([cat_condition.size(1)]).to(ref_mel.device),
                                                        ref_mel, style, None, diffusion_steps,
                                                        inference_cfg_rate=inference_cfg_rate)
                vc_target = vc_target[:, :, ref_mel.size(-1):]
                vc_wave = vocoder_fn(vc_target)[0]
            if vc_wave.ndim == 1:
                vc_wave = vc_wave.unsqueeze(0)
            if processed_frames == 0:
                if is_last_chunk:
                    output_wave = vc_wave[0].cpu().numpy()
                    generated_wave_chunks.append(output_wave) # last chunk
                    break
                output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
                generated_wave_chunks.append(output_wave) # not last so takeaway the overlapping part
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
              
            elif is_last_chunk:
                output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                processed_frames += vc_target.size(2) - overlap_frame_len
                break
              
            else:
                output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
    
    final_out = np.concatenate(generated_wave_chunks).astype(np.float32)
    sf.write('converted.wav', final_out, sr)
