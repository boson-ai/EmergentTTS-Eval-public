#!/usr/bin/env python
import gradio as gr
import json, os, datetime as dt, threading, hashlib

# ── persistent response directory ─────────────────────────────
RESP_DIR = "./survey_responses"
os.makedirs(RESP_DIR, exist_ok=True)
_WRITE_LOCK = threading.Lock()

LABELLER_PATH = "./labellers.json" 
MANIFEST_PATH = "./workloads/manifest.json"

# ---------- helpers -----------------------------------------------------------
def user_id(name, email):
    return hashlib.sha256(f"{name.strip().lower()}|{email.strip().lower()}".encode()).hexdigest()

def resp_path(name, email):
    return os.path.join(RESP_DIR, f"{user_id(name,email)}.json")

def load_labellers():
    with open(LABELLER_PATH) as f:
        data = json.load(f)
    return {entry["username"].lower(): entry["email"].lower() for entry in data}

def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)
    
def load_questions(path):
    with open(path) as f:
        return [SurveyQuestion(json.loads(l)) for l in f]

# ---------- Human‑rater instructions -----------------
HUMAN_CRITERIA_MAP = {
    "Questions": """
**Category: Questions**

**How to evaluate:**  
- Decide which recording applies correct intonation patterns: Interrogative for questions, declarative for statements, etc.
- E.g. Questions usually have a distinct pitch movement, often rising at the end in yes/no questions, while wh-questions may have a more neutral or falling tone.
- E.g. Statements between questions should have an intonation pattern that differentiates them from the questions and makes it clear that it is a statement.

**Example:** “Did you see the message? Well, I hope you did. But please tell me you actually did?”  
**Explanation:** The TTS should rise on “Did you see the message?” and “tell me you actually did?”, while “Well, I hope you did.” stays flat to sound like a statement.

*When making your judgment, focus only on the guidelines provided—do not let speaker gender, voice quality, or timbre affect your decision. If two audios have similar overall performance, try to identify any remaining differences; if none are significant and all are subtle, declare a tie.*
""",

    "Emotions": """
**Category: Emotions**

**How to evaluate:**  
- Decide which recording expresses emotions more naturally, using variations in pitch, loudness, rhythm, etc., when quoting dialogue.
- Keep the narration (non-quoted text) neutral and distinct and demonstrate tone variations between the quoted dialogues and the narrative text.

**Example:** “I can’t believe it! This is amazing!” … “Okay, wait—maybe this isn’t such a good idea after all.”  
**Explanation:** The first quote should sound joyful and energetic; the second should convey hesitation or disappointment, while the surrounding text stays calm.

*When making your judgment, focus only on the guidelines provided—do not let speaker gender, voice quality, or timbre affect your decision. If two audios have similar overall performance, try to identify any remaining differences; if none are significant and all are subtle, declare a tie.*
""",

    "Syntactic Complexity": """
**Category: Syntactic Complexity**

**How to evaluate:**  
- Decide which recording better uses prosody (pausing, phrasing, intonation, stress) to make complex sentence structures easily understandable.
- Occasionally, the text may contain homographic words, in that case, the TTS system should pronounce the homographic words with appropriate pronunciation.

**Example:** “The book that the professor who won the award wrote is on the table.”  
**Explanation:** A clear pause after “The book,” another after “award,” and stress on “book” and “is” help you track the nested clauses.

*When making your judgment, focus only on the guidelines provided—do not let speaker gender, voice quality, or timbre affect your decision. If two audios have similar overall performance, try to identify any remaining differences; if none are significant and all are subtle, declare a tie.*
""",

    "Foreign Words": """
**Category: Foreign Words**

**How to evaluate:**  
- Decide which recording pronounces foreign words more accurately (original or anglicized).
- Look for seamless transitions when code-switching between languages.

**Example:** “During his shaadi, Manoj went pura paagal and started dancing jaise he was an actor.”  
**Explanation:** The Hindi words "shaadi," "paagal," and "jaise" should sound natural, and the switch back to English should flow without a pause.

*When making your judgment, focus only on the guidelines provided—do not let speaker gender, voice quality, or timbre affect your decision. If two audios have similar overall performance, try to identify any remaining differences; if none are significant and all are subtle, declare a tie.*
""",

    "Paralinguistics": """
**Category: Paralinguistics**

**How to evaluate:**  
- Decide which recording better synthesizes speech correspond to paralinguistic cues present in the text. There can be multiple types of paralinguistic cues present in the text, like:
- Interjections (“Ugh!”, “Hmmm”), vocal sounds ("Shhh!", "Achoo!", "Meow"), emphasis using CAPS ("He didn't REALLY mean it"), vowel elongation ("Heyyyyyyy, okayyyyyyy"), hyphenation/syllable stress ("ab-so-lutely", "im-por-tant"), stuttering and hesitation ("I-I-I", "W-we-well...").

**Example:** “Ugh! I-I told you… DO NOT touch that! Seriously?!”  
**Explanation:** You should feel the frustration in “Ugh!”, the hesitation in “I-I,” the force in “DO NOT,” and the disbelief in “Seriously?!”.

*When making your judgment, focus only on the guidelines provided—do not let speaker gender, voice quality, or timbre affect your decision. If two audios have similar overall performance, try to identify any remaining differences; if none are significant and all are subtle, declare a tie.*
""",

    "Pronunciation": """
**Category: Pronunciation**

**How to evaluate:**  
- Determine which recording handles non-trival words, numerals and special characters present in the text most accurately. 
- E.g. Currency, numerals, dates, time-stamps, email addresses, passwords, urls, equations, street addresses, or tongue-twisters.

**Example:** “The equation e^(i π) + 1 = 0 is famous in mathematics.”  
**Explanation:** The system should say “e to the power of i pi plus one equals zero,” not spell out symbols or mispronounce any part.

*When making your judgment, focus only on the guidelines provided—do not let speaker gender, voice quality, or timbre affect your decision. If two audios have similar overall performance, try to identify any remaining differences; if none are significant and all are subtle, declare a tie.*
"""
}

# -----------------------------------------------------

# ---------- Custom CSS for Gradio UI -------------------
custom_css = """
/* ---------- overall page ---------- */
.gradio-container {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    background: #f9fafb;              /* subtle gray */
    color: #1f2937;                   /* dark slate   */
    padding-top: 24px;
}

/* center the outer column */
#root  > div { max-width: 900px; margin: 0 auto !important; }

/* ---------- name / email page ---------- */
input[type=text].input-textbox {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    padding: 10px 12px;
}
button.primary {                     /* Start Survey & Next buttons            */
    background: #6366f1;             /* indigo‑500                              */
    border: none;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 20px;
    transition: background 0.15s;
}
button.primary:hover:enabled { background: #4f46e5; }  /* indigo‑600 */
button.primary:disabled { opacity: 0.45; }

/* ---------- survey page ---------- */
.markdown { line-height: 1.55; }

/* tidy up the progress text */
#survey_page .markdown:first-child {
    font-weight: 600;
    margin-bottom: 4px;
}

/* place the two audio players side‑by‑side on wide screens */
@media (min-width: 640px) {
  #survey_page .audio { width: 48% !important; display: inline-block; }
}

/* radio group spacing & highlight on hover */
.radio label {
    display: block;
    padding: 8px 12px;
    margin-bottom: 6px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.12s, border 0.12s;
}
.radio input[type="radio"]:checked + label {
    background: #e0e7ff;             /* indigo ‑100   */
    border-color: #6366f1;           /* indigo ‑500   */
}
.radio label:hover { background: #f3f4f6; }

/* keep “Original text” block visually distinct */
#survey_page .markdown:nth-of-type(2) {   /* synthesize_text component */
    background: #ffffff;
    border-left: 4px solid #6366f1;
    padding: 12px 16px;
    margin-top: 12px;
    border-radius: 4px;
}

/* ---------- completion page ---------- */
#complete_page .markdown {
    font-size: 1.25rem;
    font-weight: 600;
    text-align: center;
    margin-top: 40px;
}

/* ---------- floating toast ---------- */
.toast-box {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: #4ade80;          /* emerald‑400 */
    color: white;
    padding: 12px 18px;
    border-radius: 10px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    animation: toast-fade 0.5s 2.5s forwards;  /* start fade at 2.5 s */
    z-index: 9999;              /* above everything */
}
@keyframes toast-fade {
    to { opacity: 0; transform: translateY(10px); }
}
"""
# -----------------------------------------------------

# ───────────── Survey data structures ────────────────────────────────────────
class SurveyQuestion:
    def __init__(self, entry):
        self.category = entry["category"]
        self.text_to_synthesize = f"**Original text:** {entry['text_to_synthesize']}"

        # use random order for audio A/B
        if entry["predicted_speech_index"] == 1:
            self.audio_A = entry["audio_out_path"]
            self.audio_B = entry["baseline_audio_path"]
        else:
            self.audio_A = entry["baseline_audio_path"]
            self.audio_B = entry["audio_out_path"]

        self.prompt  = HUMAN_CRITERIA_MAP[self.category]
        self.options = ["Audio A is better",
                        "Audio A and B are equal",
                        "Audio B is better"]
        self.raw = entry

class SurveySession:
    """Per‑participant state with auto‑resume."""
    def __init__(self, name, email, questions_path):
        self.questions = load_questions(questions_path)
        self.file = resp_path(name, email)
        self.responses = []
        if os.path.exists(self.file):
            try:
                with open(self.file) as f:
                    self.responses = json.load(f).get("responses", [])
            except json.JSONDecodeError:
                pass
        self.current_idx = len(self.responses)

    def current(self):
        return self.questions[self.current_idx]

    def submit(self, choice, rationale):
        q = self.current()
        # capture timestamp for this response
        timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.responses.append({
            "question_idx": self.current_idx,
            "choice":       choice,
            "rationale":    rationale,
            "question":     q.raw,
            "timestamp_utc": timestamp
        })
        self.current_idx += 1
        return self.current_idx < len(self.questions)

    def save(self, user):
        with _WRITE_LOCK, open(self.file, "w") as f:
            json.dump({
                "timestamp_utc": dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
                "user": user,
                "responses": self.responses
            }, f, indent=2)

# ──────────────── Build the Gradio UI ────────────────────────────────────────
with gr.Blocks(css=custom_css) as demo:
    toast        = gr.HTML(visible=False)
    survey_state = gr.State()
    user_info    = gr.State()

    # 0. login page
    with gr.Column() as login_pg:
        name_in = gr.Textbox(label="*Username")
        mail_in = gr.Textbox(label="*Email address")
        start_b = gr.Button("Resume / Start Survey", interactive=False)

    # 1. survey page
    with gr.Column(visible=False) as survey_pg:
        prog_m   = gr.Markdown()
        aud_A    = gr.Audio(label="Audio A")
        aud_B    = gr.Audio(label="Audio B")
        text_m   = gr.Markdown()
        prompt_m = gr.Markdown()
        radio    = gr.Radio([])
        rationale = gr.Textbox(label="Your reasoning", placeholder="(Optional) A brief rationale for your choice", lines=3)
        save_b = gr.Button("Save & Next", interactive=False, variant="primary")

    # 2. completion
    with gr.Column(visible=False) as done_pg:
        done_m = gr.Markdown("## Thank you for completing the survey!")

    # ── helpers ─────────────────────────────────────────────────────────
    def enable_start(n,e): return gr.update(interactive=bool(n.strip() and e.strip()))
    name_in.change(enable_start,[name_in,mail_in],start_b)
    mail_in.change(enable_start,[name_in,mail_in],start_b)

    # ---------- start / resume ----------------------------------------
    def start_or_resume(name,email):
        name_l  = name.strip().lower()
        email_l = email.strip().lower()

        # Check if name/email is in labeller list
        try:
            creds = load_labellers()
            valid = (name_l in creds and email_l == creds[name_l])
        except Exception:
            valid = False
        if not valid:
            ts = int(dt.datetime.utcnow().timestamp() * 1000)
            toast_html = (
                f"<div class='toast-box' id='invalid-{ts}' "
                "style='background:#dc2626;'>❌ Invalid credentials</div>"
            )
            return (
                gr.update(visible=True),   # login_pg
                gr.update(visible=False),  # survey_pg
                gr.update(visible=False),  # done_pg
                None, None,               # aud_A, aud_B
                "", "",               # prompt_m, text_m
                gr.update(choices=[], value=None),  # radio
                "",                     # prog_m
                gr.update(value=toast_html, visible=True),  # toast
                None, None               # survey_state, user_info
            )
        # valid: load user-specific questions
        manifest = load_manifest()
        questions_path = manifest.get(name_l)
        sess = SurveySession(name, email, questions_path)
        user = {"name": name, "email": email}

        # Already done
        if sess.current_idx >= len(sess.questions):   
            return (
                gr.update(visible=False),             # login_pg
                gr.update(visible=False),             # survey_pg
                gr.update(visible=True),              # done_pg
                None, None,                          # aud_A, aud_B
                "", "",                              # prompt_m, text_m
                gr.update(choices=[]),               # radio
                gr.update(value="Survey Completed"), # <-- prog_m  (slot 9)
                gr.update(visible=False, value=""), # toast
                sess,                                # survey_state (slot 10)
                user                                 # user_info   (slot 11)
            )

        # Not done yet
        q = sess.current()
        prog = f"Question {sess.current_idx+1} / {len(sess.questions)}"
        return (gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),             # done hidden
                q.audio_A, q.audio_B,
                q.prompt, q.text_to_synthesize,
                gr.update(choices=q.options, value=None),
                prog,
                gr.update(visible=False, value=""),
                sess,user)

    start_b.click(start_or_resume,
                  [name_in, mail_in],
                  [login_pg, survey_pg, done_pg,
                   aud_A, aud_B, prompt_m, text_m,
                   radio, prog_m, toast,
                   survey_state, user_info])

    # enable Save & Next only after choice
    radio.change(lambda c: gr.update(interactive=bool(c)), radio, save_b)

    # ---------- save / next -------------------------------------------
    def save_next(choice, rationale, sess:SurveySession, user):
        has_more = sess.submit(choice, rationale)
        sess.save(user)
        saved = "✅ Response saved!"

        # make the toast html unique
        toast_html = f"<div class='toast-box' id='t{int(dt.datetime.utcnow().timestamp()*1000)}'>✅ Response saved!</div>"

        if has_more:
            q = sess.current()
            prog = f"Question {sess.current_idx+1} / {len(sess.questions)}"
            return (q.audio_A, q.audio_B,
                    q.prompt, q.text_to_synthesize,
                    gr.update(choices=q.options,value=None),    # reset radio
                    gr.update(value=""),        # clear rationale
                    gr.update(interactive=False),       # disable Save & Next
                    prog,
                    gr.update(value=toast_html, visible=True),
                    sess,
                    gr.update(),                      # keep survey_pg
                    gr.update(visible=False))         # done hidden

        # finished
        return (None,None,"","",
                gr.update(choices=[],visible=False),        # hide radio
                gr.update(value="", visible=False),        # clear & hide rationale
                gr.update(interactive=False),       # disable Save & Next
                "Survey Completed",
                gr.update(value=f"<div class='toast-box'>{saved}</div>", visible=True),
                sess,
                gr.update(visible=False),             # hide survey
                gr.update(visible=True))              # show done

    save_b.click(save_next,
                 [radio, rationale, survey_state, user_info],
                 [aud_A, aud_B, prompt_m, text_m,
                  radio, rationale, save_b, prog_m, toast,
                  survey_state, survey_pg, done_pg])

# concurrency
demo.queue(default_concurrency_limit=20, max_size=30).launch(
        server_name="0.0.0.0", server_port=7861,
        allowed_paths=["/fsx/workspace/"])