"""Landing-page HTML and CSS that recreate the RAGG mock-up in Streamlit."""
from __future__ import annotations

RED = "#920C00"
STATIC = "app/static"


def _h(s: str) -> str:
    """Strip indentation + blank lines so Streamlit's Markdown never turns HTML into a code block."""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Urbanist:wght@400;500;600;700;800&display=swap');

:root {{ --red:{RED}; --ink:#111; --muted:#6b6b6b; --line:#e9e9e9; }}
.stApp {{
  background-color:#fff;
  background-image: radial-gradient(#e4e4e4 1px, transparent 1.2px);
  background-size: 22px 22px;
}}
.stApp, .stApp :is(p,h1,h2,h3,h4,h5,h6,li,td,th,label,input,textarea,button,div,a,small,strong,em,blockquote,summary) {{
  font-family:'Urbanist', system-ui, sans-serif !important;
}}
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer, .stDeployButton,
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display:none !important; }}
[data-testid="stMainBlockContainer"], .block-container {{ padding:0 !important; max-width:100% !important; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display:none; }}
html {{ scroll-behavior:smooth; }}

/* ---------- nav ---------- */
.ragg-nav {{
  position:fixed; top:14px; left:50%; transform:translateX(-50%); z-index:1000;
  width:min(1180px, calc(100% - 32px)); display:flex; align-items:center; justify-content:space-between;
  background:rgba(255,255,255,.94); backdrop-filter:blur(10px); border-radius:999px;
  box-shadow:0 6px 28px rgba(0,0,0,.08); padding:10px 12px 10px 28px;
}}
.ragg-nav .brand {{ font-weight:700; font-size:16px; color:var(--ink); }}
.ragg-nav .links a {{ color:#333; text-decoration:none; font-size:13px; margin:0 18px; }}
.ragg-nav .links a:hover {{ color:var(--red); }}
.btn-red {{ background:var(--red); color:#fff !important; border-radius:999px; padding:9px 22px; font-size:13px; font-weight:600; text-decoration:none; }}
.btn-black {{ background:#111; color:#fff !important; border-radius:999px; padding:14px 30px; font-size:14px; font-weight:600; text-decoration:none; display:inline-block; }}
.btn-ghost {{ background:#fff; color:#111 !important; border:1.5px solid #111; border-radius:999px; padding:12.5px 30px; font-size:14px; font-weight:600; text-decoration:none; display:inline-block; }}
.btn-black:hover, .btn-ghost:hover, .btn-red:hover {{ transform:translateY(-1px); box-shadow:0 6px 16px rgba(0,0,0,.15); }}

/* ---------- hero ---------- */
.hero {{ text-align:center; padding:110px 20px 70px; }}
.hero img {{ width:min(640px, 90vw); display:block; margin:0 auto -6px; }}
.hero h1 {{ font-size:clamp(34px, 5vw, 62px); line-height:1.04; font-weight:700; letter-spacing:-1.5px; color:var(--ink); margin:0; padding:0; }}
.hero h1 .r {{ color:var(--red); }}
.hero p.sub {{ color:#222; margin:18px 0 26px; font-size:15px; }}
.hero .ctas {{ display:flex; gap:14px; justify-content:center; flex-wrap:wrap; }}

/* ---------- sections ---------- */
.sec {{ padding:90px 24px; scroll-margin-top:80px; }}
.sec.red {{ background:var(--red); color:#fff; }}
.sec.white {{ background:#fff; }}
.sec h2 {{ text-align:center; font-size:clamp(26px,3vw,36px); font-weight:600; letter-spacing:-.6px; margin:0 0 14px; padding:0; }}
.sec.red h2, .sec.red p {{ color:#fff; }}
.sec p.lead {{ text-align:center; max-width:560px; margin:0 auto; font-size:15px; line-height:1.55; }}
.photos {{ display:flex; gap:28px; justify-content:center; align-items:center; margin-top:46px; flex-wrap:wrap; }}
.photos img {{ border-radius:26px; object-fit:cover; box-shadow:0 0 0 6px rgba(255,255,255,.08), 0 18px 40px rgba(0,0,0,.35); }}
.photos .s {{ width:260px; height:330px; }}
.photos .l {{ width:300px; height:400px; }}

.line-wrap {{ max-width:1180px; margin:0 auto; position:relative; }}
.line-wrap h2 {{ text-align:left; }}
.line-wrap .does {{ max-width:540px; font-size:15px; line-height:1.55; color:#222; }}
.line-wrap .you {{ max-width:500px; margin-left:auto; font-size:15px; line-height:1.55; color:#222; }}
.line-wrap svg {{ width:100%; height:120px; display:block; }}
.line-wrap .walker {{ position:absolute; right:13%; top:70px; width:150px; }}

.how {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:30px; max-width:1180px; margin:50px auto 0; }}
.how .card {{ background:#fff; color:var(--red); border-radius:26px; padding:30px 34px 38px; box-shadow:0 18px 40px rgba(0,0,0,.25); }}
.how .top {{ display:flex; gap:18px; align-items:center; margin-bottom:22px; }}
.how .num {{ font-size:64px; font-weight:800; line-height:.8; color:#111; }}
.how .ttl {{ font-size:19px; font-weight:600; line-height:1.15; color:var(--red); }}
.how .bold {{ font-weight:700; font-size:15.5px; margin-bottom:22px; color:var(--red); }}
.how .txt {{ font-size:15px; line-height:1.55; color:var(--red); }}
.model-head {{ text-align:center; padding:90px 20px 34px; scroll-margin-top:80px; }}
.model-head h2 {{ font-size:clamp(26px,3vw,36px); font-weight:600; letter-spacing:-.6px; margin:0 0 10px; padding:0; }}
.model-head p {{ color:#222; margin:0; font-size:15px; }}

/* ---------- the model panel ---------- */
.st-key-model_wrap {{
  max-width:1180px; margin:0 auto 90px auto !important; background:#fff; border-radius:30px;
  box-shadow:0 24px 70px rgba(0,0,0,.09), 0 0 0 1px #efefef; padding:24px !important;
}}
.st-key-library h3 {{ font-size:24px; font-weight:700; margin:4px 0 0; padding:0; }}
.lib-sub {{ font-size:13px; color:#333; margin:0 0 18px; }}
.lib-label {{ display:flex; align-items:center; gap:10px; font-size:13px; font-weight:600; color:#111; margin:14px 0 6px; }}
.lib-label::after {{ content:""; flex:1; border-top:1.5px solid #cfcfcf; }}
.st-key-library .stButton button, .st-key-library [data-testid="stPopover"] button {{ width:100%; }}
.st-key-library [data-testid="stPopover"] > div > button, .st-key-library [data-testid="stPopoverButton"] {{
  background:transparent; border:none; box-shadow:none; justify-content:flex-start; padding:4px 6px; min-height:0; font-weight:600;
}}
[class*="st-key-nb_"] button {{ background:transparent; border:none; justify-content:flex-start; text-align:left; padding:5px 10px; min-height:0; border-radius:10px; font-size:13px; }}
[class*="st-key-nb_"] button:hover {{ background:#f4f4f4; }}
[class*="st-key-nbon_"] button {{ background:#111 !important; color:#fff !important; }}
[class*="st-key-nbon_"] button p {{ color:#fff !important; }}

[class*="st-key-pcard_"] {{ border-radius:14px; padding:8px 8px 6px 12px !important; border-bottom:1px solid var(--line); gap:0 !important; }}
[class*="st-key-pcard_on_"] {{ background:#111; box-shadow:0 8px 18px rgba(0,0,0,.18); border-bottom:none; margin:4px 0; }}
[class*="st-key-pcard_"] .stButton button {{
  background:transparent !important; border:none !important; box-shadow:none !important; padding:0 !important; min-height:0 !important;
  justify-content:flex-start; text-align:left;
}}
[class*="st-key-pcard_"] .stButton button p {{ font-weight:700; font-size:13px; line-height:1.25; text-align:left; }}
[class*="st-key-pcard_on_"] .stButton button p, [class*="st-key-pcard_on_"] .pc-meta, [class*="st-key-pcard_on_"] [data-testid="stPopover"] button p {{ color:#fff !important; }}
[class*="st-key-pcard_"] [data-testid="stPopover"] button {{ background:transparent; border:none; box-shadow:none; min-height:0; padding:0 4px; }}
.pc-meta {{ font-size:11px; font-weight:600; color:#555; margin:2px 0 2px; }}
.pc-warn {{ font-size:11px; color:#b06000; }}

.st-key-chat_panel {{ background:#fafafa; border:1px solid #efefef; border-radius:24px; padding:16px 16px 12px !important; }}
.st-key-chat_panel [data-testid="stVerticalBlockBorderWrapper"] {{ border:none !important; }}
.empty {{ text-align:center; padding:120px 10px 30px; }}
.empty .big {{ font-size:30px; font-weight:500; color:#c4c4c4; letter-spacing:-.5px; }}
.empty .small {{ font-size:13px; color:#c4c4c4; }}
.st-key-chips button, [class*="st-key-chip_"] button {{ border-radius:999px; border:1px solid #ddd; background:#fff; font-size:12.5px; padding:4px 12px; min-height:0; }}
.st-key-chips button:hover, [class*="st-key-chip_"] button:hover {{ border-color:var(--red); color:var(--red); }}

.st-key-ask_bar [data-testid="stForm"] {{ border:1px solid #e6e6e6; border-radius:999px; background:#fff; padding:6px 8px 6px 18px; box-shadow:0 4px 18px rgba(0,0,0,.05); }}
.st-key-ask_bar [data-testid="stTextInput"] input {{ border:none !important; background:transparent !important; box-shadow:none !important; font-size:14px; }}
.st-key-ask_bar [data-testid="stTextInput"] > div, .st-key-ask_bar [data-baseweb="input"], .st-key-ask_bar [data-baseweb="base-input"] {{ border:none !important; background:transparent !important; }}
button[kind^="primary"], [data-testid^="stBaseButton-primary"] {{ background:#111 !important; border:none !important; color:#fff !important; border-radius:999px !important; font-weight:600 !important; }}
button[kind^="primary"]:hover, [data-testid^="stBaseButton-primary"]:hover {{ background:var(--red) !important; }}
.scope-chip {{ font-size:12px; color:#555; margin:2px 4px 8px; }}
.scope-chip b {{ color:#111; }}

/* chat content */
.cite {{ background:#fbe9e7; color:var(--red); border-radius:6px; padding:1px 7px; font-size:.8em; font-weight:700; white-space:nowrap; }}
.unver {{ background:#fff3d6; color:#8a5a00; border-radius:6px; padding:1px 7px; font-size:.76em; font-weight:600; }}
.sec-tag {{ color:#888; font-size:.78em; }}
.quote {{ font-style:italic; }}
.ev-card {{ background:#fff; border:1px solid #eee; border-left:4px solid var(--red); border-radius:12px; padding:10px 14px; margin:6px 0; }}
.ev-head {{ font-size:12px; font-weight:700; color:#111; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
.ev-q {{ font-size:14px; line-height:1.5; margin-top:6px; color:#222; }}
.b-ok {{ background:#e7f6ec; color:#17703a; border-radius:6px; padding:1px 7px; font-size:11px; }}
.b-near {{ background:#fff3d6; color:#8a5a00; border-radius:6px; padding:1px 7px; font-size:11px; }}
.b-ret {{ background:#eef1f7; color:#3d4b6b; border-radius:6px; padding:1px 7px; font-size:11px; }}
[data-testid="stChatMessage"] {{ background:transparent; }}
[data-testid="stChatMessage"] table {{ font-size:13px; }}

/* ---------- FAQ + footer ---------- */
.faq {{ max-width:900px; margin:0 auto; }}
.faq details {{ background:#fff; border:1px solid #eee; border-radius:16px; padding:14px 20px; margin:10px 0; }}
.faq summary {{ font-weight:700; cursor:pointer; font-size:15.5px; }}
.faq details p, .faq details li {{ font-size:14.5px; line-height:1.55; color:#333; }}
.foot {{ background:var(--red); color:#fff; padding:44px 48px 22px; }}
.foot .cols {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:20px; border-top:1px solid rgba(255,255,255,.35); padding-top:18px; max-width:1300px; margin:0 auto; }}
.foot h5 {{ color:#fff; font-size:12px; font-weight:700; margin:0 0 10px; padding:0; }}
.foot a {{ display:block; color:#fff !important; text-decoration:none; font-size:12px; margin:5px 0; opacity:.92; }}
.foot .bottom {{ display:flex; justify-content:space-between; max-width:1300px; margin:34px auto 0; font-size:11px; }}
.foot .bottom span a {{ display:inline; margin-left:18px; font-size:11px; }}

@media (max-width: 900px) {{
  .how {{ grid-template-columns:1fr; }}
  .foot .cols {{ grid-template-columns:repeat(2, 1fr); }}
  .ragg-nav .links {{ display:none; }}
  .line-wrap .walker {{ position:static; display:block; margin:10px auto; }}
}}
</style>
"""


def landing_top() -> str:
    return _h(f"""
<div class="ragg-nav" id="top">
  <span class="brand">Ragg for Research</span>
  <span class="links"><a href="#home" target="_self">Home</a><a href="#faq" target="_self">Got Questions?</a><a href="#how" target="_self">How it works</a></span>
  <a class="btn-red" href="#ragg-model" target="_self">Get Started</a>
</div>
<div class="hero" id="home">
  <img src="{STATIC}/hero.webp" alt="Researchers working in a laboratory">
  <h1>Semantic Literature Synthesis<br>for <span class="r">Modern Researchers</span></h1>
  <p class="sub">Stop skimming hundreds of pages. Query your literature review directly</p>
  <div class="ctas">
    <a class="btn-black" href="#ragg-model" target="_self">Create Your Knowledge Space</a>
    <a class="btn-ghost" href="?demo=1#ragg-model" target="_self">Try Interactive Demo</a>
  </div>
</div>
<div class="sec red">
  <h2>The Human in Control, not AI.</h2>
  <p class="lead">It is a precision semantic indexing tool designed to surface source passages, highlight exact paragraphs, and leave every conclusion in your hands.</p>
  <div class="photos">
    <img class="s" src="{STATIC}/photo1.jpg" alt="Scientist working at a lab bench">
    <img class="l" src="{STATIC}/photo2.jpg" alt="Scientist examining a test tube">
    <img class="s" src="{STATIC}/photo3.jpg" alt="Students reading in a library">
  </div>
</div>
<div class="sec white">
  <div class="line-wrap">
    <h2>The Line We Don't Cross</h2>
    <p class="does"><b>What RAGG does:</b> Vectorizes dense literature, maps semantic relationships across your Knowledge Spaces, jumps to exact page numbers, and highlights the three most relevant raw excerpts.</p>
    <img class="walker" src="{STATIC}/walker.webp" alt="">
    <svg viewBox="0 0 1000 120" preserveAspectRatio="none"><line x1="0" y1="20" x2="1000" y2="100" stroke="#111" stroke-width="2"/></svg>
    <p class="you"><b>What you do:</b> Evaluate methodology, critique validity, interpret nuance, and construct your synthesis.</p>
  </div>
</div>
<div class="sec red" id="how">
  <h2>How Ragg for Research Works</h2>
  <p class="lead">It is a precision semantic indexing tool designed to surface source passages, highlight exact paragraphs, and leave every conclusion in your hands.</p>
  <div class="how">
    <div class="card"><div class="top"><span class="num">1</span><span class="ttl">Source Verification<br>at a Glance</span></div>
      <div class="bold">Generic LLMs invent facts when they can't find answers.</div>
      <div class="txt">RAGG pairs every result with an active viewport into your actual PDF, displaying the original typography and exact context so you never cite a claim without seeing it in print.</div></div>
    <div class="card"><div class="top"><span class="num">2</span><span class="ttl">An Index,<br>Not an Author</span></div>
      <div class="bold">Research ethics demand traceable lineage.</div>
      <div class="txt">RAGG acts as an accelerated magnifying glass across your reading stack—empowering your critical thinking instead of automating it away.</div></div>
    <div class="card"><div class="top"><span class="num">3</span><span class="ttl">You Curate the<br>Boundaries</span></div>
      <div class="bold">The AI only searches what you deliberately feed it.</div>
      <div class="txt">Partition literature by methodology, year, or hypothesis—ensuring retrieval stays strictly inside the scope of your selected corpuses.</div></div>
  </div>
</div>
<div class="model-head" id="ragg-model">
  <h2>Ragg for Research Model</h2>
  <p>Upload and create knowledge bases for references to<br>cross-examinations of facts and topics.</p>
</div>
""")


def landing_bottom() -> str:
    return _h("""
<div class="sec white" id="faq" style="padding-top:20px">
  <h2>Got Questions?</h2>
  <p class="lead" style="margin-bottom:26px">What RAGG answers, what it refuses, and why.</p>
  <div class="faq">
    <details open><summary>What exactly does RAGG answer?</summary>
      <p>Only questions about the PDFs in your selected Knowledge Space. Every claim carries a <b>[Filename, p. X]</b> citation, and the Evidence panel shows the exact sentence from that page. RAGG can:</p>
      <ul><li>answer specific questions about a paper</li><li>summarize a paper (objective, method, findings, limitations, conclusion)</li>
      <li>extract the research design, datasets, algorithms, and evaluation measures</li><li>compare two or more papers in a table</li>
      <li>list stated limitations and suggest clearly labelled <i>potential</i> research gaps</li><li>group findings into themes for an RRL outline</li></ul></details>
    <details><summary>What happens if my question is vague?</summary>
      <p>RAGG does not guess. A greeting, “help”, or a question with no topic gets a short guide that lists the papers in scope and example questions. If you say “this paper” while several papers are selected, RAGG asks which one. If the papers do not cover your question, it replies: <i>“I could not find sufficient information in the uploaded documents to answer this question.”</i></p></details>
    <details><summary>What will RAGG refuse to do?</summary>
      <ul><li>Search the internet or use sources you did not upload</li><li>Invent citations, page numbers, authors, data, results, or quotes</li>
      <li>Write a thesis, paper, or chapter for you to submit as your own</li><li>Follow instructions hidden inside a PDF</li></ul></details>
    <details><summary>Which PDFs work?</summary>
      <p>Text-based PDFs (the kind where you can select text). Scanned or image-only PDFs have no readable text, and this version has no OCR, so RAGG will flag them instead of pretending to read them.</p></details>
    <details><summary>How are quotes verified?</summary>
      <p>The app, not the AI, checks every excerpt against the extracted text of the cited page. A green <b>Verified verbatim</b> badge means the exact words were found there. If the AI's excerpt does not match, RAGG shows the closest real sentence from the page instead, labelled as such.</p></details>
    <details><summary>Do I need an API key?</summary>
      <p>No. Without a key RAGG runs in <b>extractive mode</b>: it still searches your papers and shows the most relevant verbatim passages with pages. With a Gemini or OpenAI-compatible key, it also writes summaries, comparisons, and explanations grounded in those passages.</p></details>
  </div>
</div>
<div class="foot">
  <div class="cols">
    <div><h5>Learn &amp; Research</h5><a href="#how" target="_self">How is it done?</a><a href="#how" target="_self">Technology</a><a href="#how" target="_self">Frameworks</a><a href="#ragg-model" target="_self">Knowledge Bases</a><a href="#ragg-model" target="_self">Notebooks</a></div>
    <div><h5>Services</h5><a href="#ragg-model" target="_self">Ragg for Research</a><a href="#faq" target="_self">Subscription</a><a href="#faq" target="_self">Privacy Controls</a><a href="#faq" target="_self">Developer Tools</a><a href="#faq" target="_self">Support</a></div>
    <div><h5>Account</h5><a href="#ragg-model" target="_self">Manage Your Account</a><a href="#faq" target="_self">Plan Usage</a><a href="#faq" target="_self">Warranty</a><a href="#faq" target="_self">Pricing</a><a href="#faq" target="_self">FAQs</a></div>
    <div><h5>About Ragg for Research</h5><a href="#home" target="_self">Overview</a><a href="#home" target="_self">Newsroom</a><a href="#how" target="_self">Development</a><a href="#how" target="_self">Research</a><a href="#faq" target="_self">Contact</a></div>
  </div>
  <div class="bottom"><span>© 2026 RAGG Inc. All rights reserved.</span><span><a href="#faq" target="_self">Privacy Policy</a><a href="#faq" target="_self">Terms of Use</a><a href="#faq" target="_self">Legal</a><a href="#top" target="_self">Site Map</a></span></div>
</div>
""")
