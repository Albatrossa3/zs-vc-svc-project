# demo_gallery.py
import os
import re
import gradio as gr

DEMO_DIR = os.path.join(os.getcwd(), "conversion_demo")
VC_DIR = os.path.join(DEMO_DIR, "vc")
SVC_DIR = os.path.join(DEMO_DIR, "svc")

def _collect_indices(dir_path: str, prefix: str):
    # parse indices from filenames like: prefix-source-2.wav, prefix-ref-1.wav, prefix-2-1.wav
    src_pat = re.compile(rf"^{re.escape(prefix)}-source-(\d+)\.wav$", re.IGNORECASE)
    ref_pat = re.compile(rf"^{re.escape(prefix)}-ref-(\d+)\.wav$", re.IGNORECASE)
    res_pat = re.compile(rf"^{re.escape(prefix)}-(\d+)-(\d+)\.wav$", re.IGNORECASE)

    src_ids, ref_ids, pairs = set(), set(), set()
    if not os.path.isdir(dir_path):
        return sorted(src_ids), sorted(ref_ids), sorted(pairs)
    for fn in os.listdir(dir_path):
        m = src_pat.match(fn)
        if m:
            src_ids.add(int(m.group(1)))
            continue
        m = ref_pat.match(fn)
        if m:
            ref_ids.add(int(m.group(1)))
            continue
        m = res_pat.match(fn)
        if m:
            pairs.add((int(m.group(1)), int(m.group(2))))
    return sorted(src_ids), sorted(ref_ids), sorted(pairs)

def list_available(mode: str):
    if mode == "VC":
        return _collect_indices(VC_DIR, "vc")
    return _collect_indices(SVC_DIR, "svc")

def get_demo_files(mode: str, source_id: int, ref_id: int):
    if mode == "VC":
        base = VC_DIR
        prefix = "vc"
    else:
        base = SVC_DIR
        prefix = "svc"
    src = os.path.join(base, f"{prefix}-source-{source_id}.wav")
    ref = os.path.join(base, f"{prefix}-ref-{ref_id}.wav")
    res = os.path.join(base, f"{prefix}-{source_id}-{ref_id}.wav")
    # handle missing files
    return (
        src if os.path.exists(src) else None,
        ref if os.path.exists(ref) else None,
        res if os.path.exists(res) else None,
    )

def refresh_choices(mode: str):
    src_ids, ref_ids, pairs = list_available(mode)
    src_choices = [str(i) for i in src_ids]
    ref_choices = [str(i) for i in ref_ids]
    src_def = src_choices[0] if src_choices else None
    ref_def = ref_choices[0] if ref_choices else None
    return gr.update(choices=src_choices, value=src_def), \
           gr.update(choices=ref_choices, value=ref_def)

def update_audio(mode: str, src_sel: str, ref_sel: str):
    if not src_sel or not ref_sel:
        return None, None, None
    s, r = int(src_sel), int(ref_sel)
    return get_demo_files(mode, s, r)

def main():
    with gr.Blocks(title="Conversion Demo Gallery") as demo:
        gr.Markdown(
            "**Zero-shot Conversion Demo Gallery**  \n"
            "Select VC or SVC, pick a source index and a reference index to see the converted result."
        )
        with gr.Row():
            mode = gr.Radio(choices=["VC", "SVC"], value="VC", label="Mode")

        # initialize dropdowns from filesystem
        init_src_ids, init_ref_ids, _ = list_available("VC")
        init_src_choices = [str(i) for i in init_src_ids]
        init_ref_choices = [str(i) for i in init_ref_ids]
        init_src_val = init_src_choices[0] if init_src_choices else None
        init_ref_val = init_ref_choices[0] if init_ref_choices else None

        with gr.Row():
            src_sel = gr.Dropdown(choices=init_src_choices, value=init_src_val, label="Source index")
            ref_sel = gr.Dropdown(choices=init_ref_choices, value=init_ref_val, label="Reference index")

        with gr.Row():
            src_audio = gr.Audio(label="Source", type="filepath")
            ref_audio = gr.Audio(label="Reference", type="filepath")
            res_audio = gr.Audio(label="Converted Result", type="filepath")

        def _on_mode_change(m):
            return refresh_choices(m)

        mode.change(
            fn=_on_mode_change,
            inputs=[mode],
            outputs=[src_sel, ref_sel],
        )

        def _on_any_change(m, s, r):
            return update_audio(m, s, r)

        # Update audio on any selection change
        mode.change(_on_any_change, [mode, src_sel, ref_sel], [src_audio, ref_audio, res_audio])
        src_sel.change(_on_any_change, [mode, src_sel, ref_sel], [src_audio, ref_audio, res_audio])
        ref_sel.change(_on_any_change, [mode, src_sel, ref_sel], [src_audio, ref_audio, res_audio])

        demo.load(_on_any_change, [mode, src_sel, ref_sel], [src_audio, ref_audio, res_audio])

    demo.launch()

if __name__ == "__main__":
    main()