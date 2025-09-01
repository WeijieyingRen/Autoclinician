# eval_dialogue.py  (semantic-only; robust generation + JSON-only parse)
import os, re, json, argparse, math
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
import torch
from tqdm import tqdm


from metrics_semantic import (
    BioEmbedMatcher,
    hit_at_k_sem, recall_at_k_sem, mrr_sem, ndcg_at_k_sem,
    f1_semantic, set_acc_semantic,
    micro_f1_semantic, macro_f1_semantic
)

# ---------------- JSON helpers ----------------
ANCHOR_RE = re.compile(r"\bjson\s*:\s*", re.I)

def _segment_after_json_anchor(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    last = None
    for m in ANCHOR_RE.finditer(text):
        last = m
    if last:
        return text[last.end():]
    i = text.find("{")
    return text[i:] if i != -1 else text

def _extract_first_json_object(s: str) -> Optional[str]:
    if not s: return None
    i = s.find("{")
    if i < 0: return None
    depth = 0
    for j, ch in enumerate(s[i:], start=i):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[i:j+1]
    return None

def dedup_keep_order(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        t = str(x).strip()
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

def split_pipe(s: str) -> List[str]:
    if s is None: return []
    s = str(s).strip()
    return [t.strip() for t in s.split("|") if t.strip()] if s else []

def parse_predictions(text: str) -> Tuple[List[str], List[str]]:

    seg = _segment_after_json_anchor(text or "")
    obj_str = _extract_first_json_object(seg)
    if not obj_str:
        return [], []
    try:
        obj = json.loads(obj_str)
    except Exception:
        return [], []
    tests = obj.get("tests", []) or obj.get("lab_tests", [])
    diags = obj.get("diagnosis", []) or obj.get("diagnoses", [])
    if isinstance(tests, str):  tests = split_pipe(tests)
    if isinstance(diags, str):  diags = split_pipe(diags)
    return dedup_keep_order([str(x) for x in tests]), dedup_keep_order([str(x) for x in diags])

# ---------------- Build rolling context (clean assistant leaks) ----------------
CLEAN_SCHEDULE_RE = re.compile(r"(please\s+schedule.*?$|schedule\s*:\s*.*?$)", re.I|re.S)
CLEAN_DIAG_RE     = re.compile(r"(likely diagnoses?\s*:\s*.*?$|diagnoses?\s*:\s*.*?$)", re.I|re.S)

def sanitize_assistant_text(s: str) -> str:
    if not isinstance(s, str): return ""
    t = CLEAN_SCHEDULE_RE.sub("", s)
    t = CLEAN_DIAG_RE.sub("", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()

def build_history_before_user(msgs, upto_user_idx: int) -> str:
    parts = []
    for k in range(upto_user_idx):  # exclude current user turn
        mk = msgs[k]; role = mk.get("role"); txt = mk.get("content", "")
        if not txt: continue
        if role == "user":
            parts.append(f"[User]\n{txt}")
        elif role == "assistant":
            clean = sanitize_assistant_text(txt)
            if clean: parts.append(f"[Assistant]\n{clean}")
    return "\n\n".join(parts).strip()

def clip_by_chars(s: str, max_chars: int = 4000) -> str:
    return s if not isinstance(s, str) or len(s) <= max_chars else s[-max_chars:]

# ---------------- Gold extraction (from assistant messages) ----------------
TEST_LIST_SPLIT = re.compile(r"[\n;,、，•\-]+")

def extract_tests_from_assistant(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip(): return []
    m = re.search(r"(please\s+schedule(?:\s+for\s+your\s+next\s+visit)?|schedule)\s*[:\-]\s*(.+)$", text, re.I|re.S)
    chunk = m.group(2) if m else None
    if not chunk:
        m2 = re.search(r"please\s+schedule(?:\s+for\s+your\s+next\s+visit)?\s*(.+)$", text, re.I|re.S)
        chunk = m2.group(1) if m2 else None
    if not chunk: return []
    items, out = TEST_LIST_SPLIT.split(chunk), []
    for p in items:
        t = p.strip(" .:\t")
        if not t: continue
        if re.search(r"\b(reassess|we will|once the results|next visit)\b", t, re.I): break
        if len(t) > 120: continue
        out.append(t[:-1] if t.endswith(",") else t)
    return dedup_keep_order(out)

def parse_final_diagnoses_from_text(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip(): return []
    m = re.search(r"(likely diagnoses?|diagnoses?)\s*[:\-]\s*(.+)$", text, re.I|re.S)
    if not m: return []
    tail = m.group(2).split("\n")[0]
    return dedup_keep_order([p.strip(" .") for p in re.split(r"[;,|]+", tail) if p.strip()])

# ---------------- Flatten an episode to rows ----------------
def flatten_episode(ep: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    One row per assistant turn.
      prompt_text = (history up to previous user) + current user text
      gold_tests  = schedule list in this assistant turn
      gold_diag   = only on the LAST assistant turn (labels preferred)
    """
    rows: List[Dict[str, Any]] = []
    ep_id = ep.get("meta", {}).get("episode_id") or ep.get("meta", {}).get("subject_id") or "episode"
    msgs = ep.get("messages", []) or []
    labels = ep.get("labels", {}) or {}
    final_titles = [str(x).strip() for x in (labels.get("final_diagnoses_title") or []) if str(x).strip()]

    turn_idx = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue

        # locate the most recent user before this assistant
        j = i - 1
        prompt_current = ""
        while j >= 0:
            if msgs[j].get("role") == "user":
                prompt_current = msgs[j].get("content", "") or ""
                break
            j -= 1

        history = build_history_before_user(msgs, upto_user_idx=j) if j >= 0 else ""
        prompt_text = f"{history}\n\n[Current User]\n{prompt_current}" if history else prompt_current
        prompt_text = clip_by_chars(prompt_text, max_chars=4000)

        asst_text = m.get("content", "") or ""
        gold_tests = extract_tests_from_assistant(asst_text)

        gold_diag = ""
        if i == len(msgs) - 1 or all(x.get("role") != "assistant" for x in msgs[i+1:]):
            gold_diag = "|".join(final_titles) if final_titles else "|".join(parse_final_diagnoses_from_text(asst_text))

        rows.append({
            "dialogue_id": ep_id,
            "turn_index": turn_idx,
            "prompt_text": prompt_text,
            "gold_tests": "|".join(gold_tests),
            "gold_diagnosis": gold_diag
        })
        turn_idx += 1
    return rows

def load_table(path: str) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".jsonl") or p.endswith(".ndjson"):
        recs: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    ep = json.loads(line)
                except Exception:
                    continue
                recs.extend(flatten_episode(ep))
        return pd.DataFrame(recs)
    return pd.read_csv(path)

# ---------------- Generation helpers ----------------
def load_hf_model(model_path: str, device_str: str):
    """稳定加载 HF 模型；返回 (tokenizer, model, device or None)。"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # 方式A：手动指定设备（不再使用 device_map）
        device = torch.device(device_str)
        dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True
        ).eval().to(device)

        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        return tok, model, device
    except Exception as e:
        print(f"[WARN] HF model load failed -> fall back to pred_text. {e}")
        return None, None, None

def build_chat_prompt(tok, system_msg: str, user_msg: str) -> str:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
        # 结尾给出 JSON 锚，便于模型和解析对齐：
        {"role": "system", "content": "JSON:"}
    ]
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        return f"[SYSTEM]\n{system_msg}\n\n[USER]\n{user_msg}\n\nJSON:\n"

def generate_pred_text(prompt_text: str,
                       tok,
                       model,
                       device,
                       system_prompt: str,
                       max_new_tokens: int):
    if not prompt_text or tok is None or model is None:
        return ""

    prompt = build_chat_prompt(tok, system_prompt, prompt_text)
    inputs = tok(prompt, return_tensors="pt")
    if device is not None:
        
        inputs = {k: v.to(device) for k, v in inputs.items()}

    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

    try:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            repetition_penalty=1.0
        )
        text = tok.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[WARN] generation failed: {e}")
        text = ""
    return text

def debug_sim_matrix(matcher, preds, golds, topn=3):
    from metrics_semantic import _basic_norm  # 或者把 _basic_norm 复制放本文件
    P = [ _basic_norm(p) for p in preds ]
    G = [ _basic_norm(g) for g in golds ]
    E = matcher._emb(P + G)
    Pvec, Gvec = E[:len(P)], E[len(P):]
    S = Pvec @ Gvec.T  # 余弦
    for i, p in enumerate(preds):
        top = sorted([(float(S[i,j]), golds[j]) for j in range(len(golds))], reverse=True)[:topn]
        print(f"- {p}: " + ", ".join([f"{name} ({score:.2f})" for score, name in top]))

NAN = float("nan")

def per_turn_metrics(
    gold: List[str],
    pred: List[str],
    kind: str,          # "test" or "diag"
    prefix: str,        # "lab"  or "diag"
    matcher,
    hit_ks=(1, 3, 5, 10, 15),
    recall_ks=(5, 10, 15),
    ndcg_ks=(5, 10, 15),
):
    if not gold:
        out = {f"{prefix}_mrr": float("nan"),
               f"{prefix}_f1": float("nan"),
               f"{prefix}_set_acc": float("nan")}
        for k in hit_ks:    out[f"{prefix}_hit@{k}"] = float("nan")
        for k in recall_ks: out[f"{prefix}_recall@{k}"] = float("nan")
        for k in ndcg_ks:   out[f"{prefix}_ndcg@{k}"] = float("nan")
        return out, float("nan")

    # Hit@K（语义）
    out = {}
    for k in hit_ks:
        out[f"{prefix}_hit@{k}"] = hit_at_k_sem(gold, pred, k, kind, matcher)

    # 其他指标（语义；贪心 1:1）
    out[f"{prefix}_mrr"]      = mrr_sem(gold, pred, kind, matcher)
    out[f"{prefix}_f1"]       = f1_semantic(pred, gold, kind, matcher)
    out[f"{prefix}_set_acc"]  = set_acc_semantic(gold, pred, kind, matcher)

    for k in recall_ks:
        out[f"{prefix}_recall@{k}"] = recall_at_k_sem(gold, pred, k, kind, matcher)
    for k in ndcg_ks:
        out[f"{prefix}_ndcg@{k}"]   = ndcg_at_k_sem(gold, pred, k, kind, matcher)

    return out, out[f"{prefix}_f1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="/home/wjyren/AutoClinician/data/metabolic_sample.jsonl",
                        help="Flattened CSV or raw JSONL of episodes")
    parser.add_argument("--model_path", type=str, default="/home/wjyren/AutoClinician/code/models/Qwen2.5-7B-Instruct",
                        help="HF path. If absent or load fails, use pred_text column.")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="/home/wjyren/AutoClinician/code/results")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--th_test", type=float, default=0.4)
    parser.add_argument("--th_diag", type=float, default=0.4)
    parser.add_argument("--max_pred_tests", type=int, default=100)
    parser.add_argument("--max_pred_diags", type=int, default=50)
    parser.add_argument("--infer_max_rounds", type=int, default=15,
                        help="每个 dialogue 最多推理的回合数；超过则视为对话结束。")
    parser.add_argument("--log_every", type=int, default=1,
                    help="turn number to log")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_table(args.data_csv).sort_values(by=["dialogue_id", "turn_index"]).reset_index(drop=True)

    # 状态：对话是否结束 & 已计数的模型回合数
    round_count: Dict[str, int] = {}
    dialogue_done: Dict[str, bool] = {}
    ended_by: Dict[str, str] = {}

    # 生成器（可选）
    tok, model, device = load_hf_model(args.model_path, args.device)

    PROMPT_SYSTEM = (
        'You are a clinical assistant. Return ONLY a valid JSON object with keys "tests" and "diagnosis". '
        'Requirements: '
        '(1) "tests" must be an ORDERED list (most important first) of 3–10 initial lab/diagnostic test NAMES '
        '(no symptoms, procedures, or notes). '
        '(2) "diagnosis" must be an ORDERED list (most likely first) of candidate diagnoses; it can be empty if truly insufficient evidence. '
        '(3) If any symptoms/history are present, ALWAYS propose initial tests (e.g., vitals, CBC, BMP/CMP, glucose, urinalysis, ECG/imaging as appropriate). '
        '(4) Do NOT copy phrases from the patient message unless they are actual test names or diagnosis names. Generate diagnosis in a ICD-10 format. For example. Nontraumatic subarachnoid hemorrhage' 
        '(5) Output ONLY the JSON object (no commentary, no markdown, no code fences).'
    )

    matcher = BioEmbedMatcher(model_name=args.embed_model, th_test=args.th_test, th_diag=args.th_diag)

    rows = []
    lab_f1_turns, diag_f1_turns = [], []
    all_lab_preds, all_lab_golds = [], []
    all_diag_preds, all_diag_golds = [], []

    for idx, r in tqdm(df.iterrows(), total=len(df), desc="Inference", unit="turn"):

        dialogue_id = r.get("dialogue_id")
        turn_index  = int(r.get("turn_index", 0))
        prompt_text = str(r.get("prompt_text", "") or "")
        gold_tests      = split_pipe(r.get("gold_tests", ""))
        gold_diag_list  = split_pipe(r.get("gold_diagnosis", ""))

        # 跳过已结束对话
        if dialogue_done.get(dialogue_id, False):
            continue

        # 超过最大回合数 -> 标记结束并跳过
        if round_count.get(dialogue_id, 0) >= args.infer_max_rounds:
            dialogue_done[dialogue_id] = True
            ended_by[dialogue_id] = "max_rounds"
            continue

        # 预测文本：优先使用 pred_text；否则调用模型生成
        if "pred_text" in df.columns and isinstance(r.get("pred_text"), str) and r.get("pred_text").strip():
            pred_text = r.get("pred_text")
        else:
            pred_text = generate_pred_text(
                prompt_text=prompt_text,
                tok=tok, model=model, device=device,
                system_prompt=PROMPT_SYSTEM,
                max_new_tokens=args.max_new_tokens
            )

        pred_tests_ranked, pred_diags_ranked = parse_predictions(pred_text or "")
        if len(pred_tests_ranked) > args.max_pred_tests:
            pred_tests_ranked = pred_tests_ranked[:args.max_pred_tests]
        if len(pred_diags_ranked) > args.max_pred_diags:
            pred_diags_ranked = pred_diags_ranked[:args.max_pred_diags]

        # —— 计算 per-turn 指标（Labs / Diags）——
        lab_metrics, lab_f1  = per_turn_metrics(gold_tests,     pred_tests_ranked, "test", "lab",  matcher)
        diag_metrics, diag_f1 = per_turn_metrics(gold_diag_list, pred_diags_ranked, "diag", "diag", matcher)

        # 计数本对话使用的回合数（仅在你实际完成一次推理/评估后加 1）
        round_count[dialogue_id] = round_count.get(dialogue_id, 0) + 1

        # 若本回合产出 diagnosis -> 即刻结束该对话后续回合
        if len(pred_diags_ranked) > 0:
            dialogue_done[dialogue_id] = True
            ended_by[dialogue_id] = "diagnosis_generated"

        # 记录 per-turn 明细
        row = {
            "dialogue_id": dialogue_id,
            "turn_index": turn_index,
            "prompt_text": prompt_text,
            "gold_tests": "|".join(gold_tests),
            "pred_tests": "|".join(pred_tests_ranked),
            "gold_diagnosis": "|".join(gold_diag_list),
            "pred_diagnosis": "|".join(pred_diags_ranked),
            "raw_pred_text": pred_text or "",
            "metric_mode": "semantic",
        }
        row.update(lab_metrics)
        row.update(diag_metrics)
        rows.append(row)

        # 汇总用于数据集级别的 micro-/macro-F1
        if not math.isnan(lab_f1):  lab_f1_turns.append(lab_f1)
        if not math.isnan(diag_f1): diag_f1_turns.append(diag_f1)
        if gold_tests:
            all_lab_preds.append(pred_tests_ranked); all_lab_golds.append(gold_tests)
        if gold_diag_list:
            all_diag_preds.append(pred_diags_ranked); all_diag_golds.append(gold_diag_list)
        if (idx + 1) % max(1, args.log_every) == 0:
            print(f"Inference: {idx + 1} turns")
            print (f"Current metric: {lab_metrics}, {diag_metrics}")

    # ===== 保存 per-turn 明细 =====
    per_turn_df = pd.DataFrame(rows)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    per_turn_csv = os.path.join(out_dir, "per_turn_metrics.csv")
    per_turn_df.to_csv(per_turn_csv, index=False)

    # ===== Turn-level（按 turn_index 跨对话取均值）=====
    metric_cols = [c for c in per_turn_df.columns if c.startswith(("lab_", "diag_"))]
    turn_agg = per_turn_df.groupby("turn_index")[metric_cols].mean(numeric_only=True).reset_index()
    # 附带样本数（有 gold 的才算）
    turn_agg["n_with_lab_gold"]  = per_turn_df.groupby("turn_index")["gold_tests"].apply(lambda s: (s.str.len() > 0).sum()).values
    turn_agg["n_with_diag_gold"] = per_turn_df.groupby("turn_index")["gold_diagnosis"].apply(lambda s: (s.str.len() > 0).sum()).values
    turn_agg_csv = os.path.join(out_dir, "turn_agg_by_turn.csv")
    turn_agg.to_csv(turn_agg_csv, index=False)

    # ===== Episode-level Macro-F1（每个 dialogue 内部对各回合 F1 取均值）=====
    def _mean_safe(x): return float(pd.to_numeric(x, errors="coerce").mean())
    epi = per_turn_df.groupby("dialogue_id").agg({
        "lab_f1":  _mean_safe,
        "diag_f1": _mean_safe,
    }).rename(columns={
        "lab_f1":  "episode_lab_macro_f1",
        "diag_f1": "episode_diag_macro_f1",
    }).reset_index()
    epi_csv = os.path.join(out_dir, "episode_metrics.csv")
    epi.to_csv(epi_csv, index=False)

    # ===== Dataset-level summary =====
    summary: Dict[str, Any] = {}
    # 数据集 micro-/macro-F1
    if all_lab_golds:
        summary["dataset_lab_micro_f1"] = micro_f1_semantic(all_lab_preds, all_lab_golds, "test", matcher)
        summary["dataset_lab_macro_f1"] = macro_f1_semantic(lab_f1_turns)
    if all_diag_golds:
        summary["dataset_diag_micro_f1"] = micro_f1_semantic(all_diag_preds, all_diag_golds, "diag", matcher)
        summary["dataset_diag_macro_f1"] = macro_f1_semantic(diag_f1_turns)
    # 每列的整体均值（per-turn 视角）
    for col in metric_cols:
        summary[f"mean_{col}"] = float(pd.to_numeric(per_turn_df[col], errors="coerce").mean())
    # Episode Macro-F1 的均值（跨对话）
    if not epi.empty:
        summary["mean_episode_lab_macro_f1"]  = float(pd.to_numeric(epi["episode_lab_macro_f1"],  errors="coerce").mean())
        summary["mean_episode_diag_macro_f1"] = float(pd.to_numeric(epi["episode_diag_macro_f1"], errors="coerce").mean())

    # 保存 summary
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Saved] Per-turn metrics  -> {per_turn_csv}")
    print(f"[Saved] Turn agg (by turn_index) -> {turn_agg_csv}")
    print(f"[Saved] Episode metrics -> {epi_csv}")
    print(f"[Saved] Summary         -> {summary_path}")

if __name__ == "__main__":
    main()