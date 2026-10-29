import os, json, re, ast, csv
from collections import defaultdict

# ==== Paths (edit as needed) ====
ROOT = os.path.expanduser("~/LLaMA-Factory")
DATA_DIR = os.path.join(ROOT, "data")
TEST_JSON = os.path.join(DATA_DIR, "test.json")   # gold
PRED_JSONL = os.path.join(ROOT, "saves/Qwen2.5-Coder-1.5B/lora/predict_my_test/predictions.jsonl")  # pred
REPORT_CSV = os.path.join(ROOT, "saves/eval_report.csv")

# ==== Helpers ====
def load_array_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                arr.append(json.loads(line))
    return arr

def parse_list_string(s):
    """Parse "['Final Answer: 72']" into 'Final Answer: 72'; on failure return the string unchanged."""
    if s is None:
        return ""
    s = str(s)
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list) and v:
            return str(v[0])
    except Exception:
        pass
    return s

def clean_final_prefix(s):
    """Strip a 'Final Answer:' / 'Answer:' prefix"""
    return re.sub(r'^\s*(Final\s*Answer|Answer)\s*:\s*', '', str(s), flags=re.I).strip()

def latex_cleanup(s):
    s = str(s).replace("\\boxed", "")
    s = s.replace("\\,", "").replace(",", "")     # drop thousands separators
    return s

def to_number_str(s):
    """
    Take the last number (optional sign, decimal, or integer). Extend this if you need fractions.
    Return a string so comparisons stay stable ('72' vs '72.0' can be normalized later).
    """
    s = latex_cleanup(s)
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else None

def normalize_num_str(ns):
    """Normalize '72.0' to '72'; drop leading zeros and similar details so comparisons use one form."""
    if ns is None:
        return None
    if "." in ns:
        try:
            v = float(ns)
            if abs(v - int(v)) < 1e-9:
                return str(int(round(v)))
            return str(v).rstrip("0").rstrip(".")
        except Exception:
            return ns.strip()
    # plain integer
    return ns.lstrip("0") or "0"

def get_pred_text(obj):
    """Accept several prediction field names"""
    for k in ("prediction", "text", "response", "output", "generated_text"):
        if k in obj:
            return str(obj[k])
    if "predictions" in obj and obj["predictions"]:
        return str(obj["predictions"][0])
    # fallback
    return str(obj)

def strip_prompt_prefix(instruction, prediction):
    """
    If prediction starts with the question stem, cut that prefix so numbers in the stem are not taken as the answer.
    Loose match: ignore leading/trailing whitespace and newline differences.
    """
    i = (instruction or "").strip()
    p = (prediction or "").strip()
    # Plain startswith; add a fuzzier match if you need it
    return p[len(i):].lstrip() if i and p.startswith(i) else p

# ==== Load gold ====
if not os.path.exists(TEST_JSON):
    raise FileNotFoundError(f"Gold test set not found: {TEST_JSON}")

gold_rows = load_array_json(TEST_JSON)
gold = []
for ex in gold_rows:
    instr = (ex.get("instruction") or "").strip()
    fa0 = parse_list_string(ex.get("final_answer"))
    fa  = clean_final_prefix(fa0)
    gnum = normalize_num_str(to_number_str(fa))
    gold.append({
        "instruction": instr,
        "gold_text": fa,
        "gold_num": gnum,
    })

# ==== Load predictions ====
if not os.path.exists(PRED_JSONL):
    raise FileNotFoundError(f"Prediction file not found: {PRED_JSONL}")

pred_rows = load_jsonl(PRED_JSONL)
# Ordered list (align in order first)
pred_seq = [( (r.get("instruction") or "").strip(), get_pred_text(r) ) for r in pred_rows]
# also build a map for fallback
pred_map = defaultdict(list)
for ins, txt in pred_seq:
    pred_map[ins].append(txt)

# ==== Align and score ====
n = len(gold)
em = 0
cov_pred = 0
cov_gold = sum(1 for g in gold if g["gold_num"] is not None)

rows = []
mismatches_preview = []
for i, g in enumerate(gold):
    instr = g["instruction"]
    gold_num = g["gold_num"]

    # Take the prediction in order; if the order runs out, pop one by instruction
    if i < len(pred_seq):
        pred_text = pred_seq[i][1]
        # If the order does not match and the instruction differs, try matching by instruction
        if pred_seq[i][0] != instr and pred_map[instr]:
            pred_text = pred_map[instr].pop(0)
    else:
        pred_text = pred_map[instr].pop(0) if pred_map[instr] else ""

    # Strip the question-stem prefix, then extract the number
    pred_cut = strip_prompt_prefix(instr, pred_text)
    pnum = normalize_num_str(to_number_str(pred_cut))

    if pnum is not None:
        cov_pred += 1

    correct = (gold_num is not None and pnum == gold_num)
    em += int(correct)

    if not correct and len(mismatches_preview) < 10:
        preview = pred_cut[:150].replace("\n", " ")
        if len(pred_cut) > 150:
            preview += "..."
        mismatches_preview.append((i, gold_num, pnum, preview))

    rows.append({
        "idx": i,
        "instruction": instr,
        "gold_text": g["gold_text"],
        "gold_num": gold_num,
        "prediction": pred_text,
        "pred_num": pnum,
        "correct": int(correct),
    })

# ==== Summary ====
print(f"Samples compared: {n}")
print(f"Gold with extractable number: {cov_gold}")
print(f"Predictions with extractable number: {cov_pred}")
acc = (em / n * 100.0) if n else 0.0
print(f"Exact-Match (numeric): {em}/{n} = {acc:.2f}%")

print("\nMismatches (up to 10):")
for idx, gnum, pnum, raw in mismatches_preview:
    print(f"[{idx}] gold={gnum} pred={pnum} | pred_raw='{raw}'")

# ==== Export a per-question report ====
os.makedirs(os.path.dirname(REPORT_CSV), exist_ok=True)
with open(REPORT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n[OK] Detailed results saved: {REPORT_CSV}")