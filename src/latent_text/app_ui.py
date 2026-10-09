"""Presentation for the local inference playground."""

from html import escape

import gradio as gr

APP_CSS = """
.gradio-container { width: 100% !important; min-width: 0 !important;
  max-width: 1240px !important; margin: auto !important;
  padding: clamp(18px, 2.4vw, 28px) clamp(14px, 2.4vw, 28px) 40px !important; }
.gradio-container > .main { padding: 0 !important; min-width: 0 !important; }
#model-panel { padding: 18px !important; border-radius: 18px !important;
  background: #141e2d !important; border: 1px solid #2a384c !important; }
#model-row { align-items: end; gap: 16px; }
#model-picker { border: 0; padding: 0; background: transparent; }
#refresh-models { flex: 0 0 144px; min-height: 44px; margin-bottom: 1px; }
#text-panels { gap: 20px; }
#text-panels .block { border-radius: 16px; }
#text-panels textarea { min-height: 210px; line-height: 1.65; font-size: 15px;
  font-family: 'Segoe UI', system-ui, sans-serif; }
#action-row { align-items: center; gap: 12px; }
#reconstruct { min-height: 48px; font-weight: 650; border-radius: 12px; }
#clear-text { min-height: 48px; border-radius: 12px; }
#examples { margin-top: 0; }
#examples .label { color: #9cacbe; }
#diagnostics { border-radius: 14px; }
#usage-note { color: #96a6ba; font-size: 12px; padding-top: 6px; }
button:focus-visible, a:focus-visible { outline: 2px solid #69dec7 !important;
  outline-offset: 3px; }
@media (max-width: 640px) {
  #model-panel { padding: 14px !important; }
  #model-row { flex-wrap: wrap; }
  #action-row { flex-direction: column; }
  #refresh-models { flex: 1 1 100%; }
  #text-panels textarea { min-height: 170px; }
}
"""

HTML_CSS = """
* { box-sizing: border-box; }
:host { font-family: 'Segoe UI', system-ui, sans-serif; color: #e4edf5; }
.hero { display: flex; align-items: center; justify-content: space-between;
  gap: 24px; padding: 4px 0 14px; }
.brand { display: flex; align-items: center; gap: 14px; }
.brand-icon { width: 46px; height: 46px; flex-shrink: 0; background: #173330;
  border: 1px solid #2a514a; border-radius: 13px; display: grid; place-items: center; }
.eyebrow { color: #70d9c3; font-size: 10px; letter-spacing: 1.8px; font-weight: 700;
  margin: 0 0 6px; text-transform: uppercase; }
h1 { font-size: clamp(23px, 3vw, 32px); letter-spacing: -1px; font-weight: 650;
  line-height: 1.2; margin: 0; color: #f0f5fa; }
.subtitle { color: #9eafc2; font-size: 14px; margin: 13px 0 0; line-height: 1.6; }
.local-badge { color: #aebfd0; border: 1px solid #2c394b; border-radius: 99px;
  padding: 8px 12px; font-size: 12px; white-space: nowrap; }
.dot { display: inline-block; background: #70d9c3; width: 6px; height: 6px;
  border-radius: 50%; margin-right: 7px; vertical-align: 1px; }
.source { display: flex; align-items: center; justify-content: space-between; gap: 16px;
  margin-top: 4px; padding: 14px 16px; border-radius: 11px;
  background: #142923; border: 1px solid #254a3e; }
.source.local { background: #182335; border-color: #2a3b51; }
.source-title { font-size: 13px; font-weight: 600; color: #b7efcf; margin: 0 0 4px; }
.local .source-title { color: #bfd7f1; }
.source-note { color: #9caeb5; font-size: 12px; margin: 0; line-height: 1.5; }
.source a { color: #9de8c2; font-size: 12px; text-decoration: none; white-space: nowrap;
  border: 1px solid #355e4c; border-radius: 7px; padding: 7px 10px; }
.source a:hover { background: #204331; text-decoration: underline; }
.specs { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.spec { font-size: 12px; color: #acbdd0; background: #111c2b; border: 1px solid #263548;
  border-radius: 7px; padding: 6px 9px; }
.spec strong { color: #e0eaf4; font-weight: 550; }
.loading { color: #9cacbe; font-size: 13px; padding: 8px 0; }
.result { border: 1px solid #29394e; border-radius: 16px; overflow: hidden;
  background: #111b29; }
.result-head { padding: 16px 20px; display: flex; align-items: center; gap: 12px;
  border-bottom: 1px solid #263447; }
.result-icon { display: grid; place-items: center; height: 32px; width: 32px;
  border-radius: 50%; background: #203145; color: #b5c8df; font-size: 16px; }
.result-title { color: #d9e7f5; font-size: 14px; font-weight: 650; margin: 0 0 3px; }
.result-note { color: #91a4ba; font-size: 12px; line-height: 1.5; margin: 0; }
.result.success .result-icon { background: #1b453b; color: #87ecd1; }
.result.success .result-title { color: #a8eed8; }
.result.different .result-icon { background: #493821; color: #f4cc8c; }
.result.different .result-title { color: #f4cc8c; }
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); padding: 18px 0; }
.metric { padding: 0 20px; border-right: 1px solid #263447; }
.metric:last-child { border-right: 0; }
.metric-label { font-size: 11px; color: #93a8bf; margin: 0 0 7px; }
.metric-value { font-size: 24px; font-weight: 600; letter-spacing: -0.5px;
  color: #e8f0f9; margin: 0; font-variant-numeric: tabular-nums; }
@media (max-width: 640px) {
  .hero { align-items: flex-start; }
  .local-badge { display: none; }
  .source { align-items: flex-start; flex-direction: column; gap: 10px; }
  .metrics { grid-template-columns: repeat(2, 1fr); gap: 22px 0; }
  .metric:nth-child(2) { border-right: 0; }
  .result-head { padding: 14px; }
  .metric { padding: 0 14px; }
}
"""

HEADER = """
<header class="hero">
  <div>
    <div class="brand">
      <div class="brand-icon" aria-hidden="true">
        <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
          <path d="M4 5h18M4 10h18M8 16h10M10 21h6" stroke="#83e6cc"
            stroke-width="2.5" stroke-linecap="round"/>
        </svg>
      </div>
      <div><p class="eyebrow">Text to vectors. Vectors to text.</p>
        <h1>Latent Text Compressor</h1></div>
    </div>
    <p class="subtitle">Compress your text into learned vectors. See how closely it comes back.</p>
  </div>
  <span class="local-badge"><span class="dot"></span>Local inference</span>
</header>
"""


def theme():
    colors = {
        "background_fill_primary": "#0c1320",
        "background_fill_secondary": "#141e2d",
        "body_background_fill": "#0c1320",
        "body_text_color": "#e2ebf5",
        "body_text_color_subdued": "#99acc2",
        "block_background_fill": "#141e2d",
        "block_border_color": "#2a384c",
        "block_label_background_fill": "#141e2d",
        "block_label_text_color": "#dbe7f5",
        "block_title_text_color": "#dbe7f5",
        "block_info_text_color": "#9aabc0",
        "input_background_fill": "#0f1927",
        "input_border_color": "#2b3b51",
        "input_border_color_focus": "#60c9b5",
        "input_placeholder_color": "#70859e",
        "button_primary_background_fill": "#69dec7",
        "button_primary_background_fill_hover": "#87ead7",
        "button_primary_border_color": "#69dec7",
        "button_primary_text_color": "#09261f",
        "button_secondary_background_fill": "#1b2a3e",
        "button_secondary_background_fill_hover": "#263a53",
        "button_secondary_border_color": "#33465e",
        "button_secondary_text_color": "#d9e7f8",
        "border_color_primary": "#2a384c",
    }
    values = {key: value for key, value in colors.items()}
    values.update({key + "_dark": value for key, value in colors.items()})
    return gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
        radius_size="lg",
        font=["Segoe UI", "system-ui", "sans-serif"],
        font_mono=["Consolas", "monospace"],
    ).set(**values, block_shadow="none", block_shadow_dark="none")


def model_details(codec, from_hf, repo):
    c = codec.model.config
    if from_hf:
        source = (
            '<div class="source"><div><p class="source-title">Weights downloaded from Hugging Face</p>'
            f'<p class="source-note">{escape(repo)} &middot; Cached on this PC. Inference runs locally.</p>'
            f'</div><a href="https://huggingface.co/{escape(repo, quote=True)}" '
            'target="_blank" rel="noopener noreferrer">View model on Hugging Face &#8599;</a></div>'
        )
    else:
        source = (
            '<div class="source local"><div><p class="source-title">Local checkpoint</p>'
            '<p class="source-note">Loaded from your saved weights on this PC.</p></div></div>'
        )
    specs = [
        f"{c.encoder_layers} encoder / {c.decoder_layers} decoder layers",
        f"{c.max_tokens:,}-token context",
        f"{c.span} tokens per vector",
        f"{c.width} features per vector",
        "GPU / CUDA" if codec.device == "cuda" else "CPU",
    ]
    return (
        source
        + '<div class="specs">'
        + "".join(f'<span class="spec"><strong>{escape(label)}</strong></span>' for label in specs)
        + "</div>"
    )


def result_summary(stats=None):
    if stats is None:
        state, icon, title = "ready", "&#8594;", "Ready to reconstruct"
        note = "Enter your text or try an example below."
        values = ["\u2014"] * 4
    else:
        exact = stats["exact_match"]
        state, icon = ("success", "&#10003;") if exact else ("different", "!")
        title = "Exact reconstruction" if exact else "Reconstruction differs"
        note = (
            "Every character matches, including whitespace."
            if exact
            else "Some characters changed. Open details below to compare them."
        )
        size = stats["vector_bytes"]
        values = [
            f"{stats['input_tokens']:,}",
            f"{stats['output_vectors']:,}",
            f"{stats['tokens_per_vector']:.1f}",
            f"{size:,} B" if size < 1024 else f"{size / 1024:.1f} KiB",
        ]
    labels = ["Input tokens", "Latent vectors", "Actual tokens / vector", "Vector data"]
    metrics = "".join(
        f'<div class="metric"><p class="metric-label">{label}</p>'
        f'<p class="metric-value">{escape(value)}</p></div>'
        for label, value in zip(labels, values)
    )
    return (
        f'<section class="result {state}" aria-live="polite"><div class="result-head">'
        f'<span class="result-icon" aria-hidden="true">{icon}</span>'
        f'<div><p class="result-title">{title}</p><p class="result-note">{note}</p></div></div>'
        f'<div class="metrics">{metrics}</div></section>'
    )
