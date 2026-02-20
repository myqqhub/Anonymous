# rebuttal_fewshot.py  
# ============================================================
# Rebuttal Experiment: Few-shot LLM Baseline Evaluation
# 0-shot / 1-shot / 3-shot 
# ============================================================

import sys
import json
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

CODE_DIR = Path("Path/code")
sys.path.insert(0, str(CODE_DIR))

from config import LLM_CONFIG
from llm_provider import get_provider

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("rebuttal")



DATA_PATH = Path("./data/training_data_5a.parquet")

COL_LABEL        = "label"
COL_DOC_TEXT     = "doc_text"
COL_PAT_TEXT     = "pat_text"
COL_FDA_ID       = "fda_id"
COL_SCORE_ENTITY = "score_entity"  

NUM_TEST    = 60
RANDOM_SEED = 42
SHOT_LIST   = [0, 1, 3]

LINKING_PROMPT_PATH = CODE_DIR / "prompts" / "prompt_linking.txt"

# ============================================================
# GPT Provider
# ============================================================

class GPTProvider:
    def __init__(self):
        self.api_key  = "yourkey"
        self.model    = "gpt-4-turbo"
        self.base_url = "URL"
        self.headers  = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _make_api_call(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 256
        }
        for attempt in range(3):
            try:
                resp = requests.post(
                    self.base_url, headers=self.headers,
                    json=payload, timeout=60
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                wait = 5 * (2 ** attempt)
                logger.warning(f"GPT attempt {attempt+1} failed: {e}. Retry {wait}s")
                time.sleep(wait)
        return ""

# ============================================================
# Prompt 
# ============================================================

def load_template(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def fill_prompt(template: str, doc_text: str, pat_text: str,
                entity_score: float) -> str:
    hint = (f"Ontology entity overlap score: {entity_score:.1f} "
            f"({'high' if entity_score > 10 else 'low'} functional overlap)")
    p = template
    p = p.replace("{device_name}",          doc_text[:400])
    p = p.replace("{device_entities_str}",   hint)
    p = p.replace("{patent_title}",          pat_text[:400])
    p = p.replace("{patent_entities_str}",   "See entity overlap score above.")
    return p

def build_few_shot_block(pos_pool: pd.DataFrame, neg_pool: pd.DataFrame,
                         k_shot: int, exclude_fda_id) -> str:
    if k_shot == 0:
        return ""

    n_pos = 1 if k_shot == 1 else 2
    n_neg = k_shot - n_pos

    pool_p = pos_pool[pos_pool[COL_FDA_ID] != exclude_fda_id]
    pool_n = neg_pool[neg_pool[COL_FDA_ID] != exclude_fda_id]
    n_pos  = min(n_pos, len(pool_p))
    n_neg  = min(n_neg, len(pool_n))

    sampled_p = pool_p.sample(n_pos, random_state=RANDOM_SEED) if n_pos > 0 else pd.DataFrame()
    sampled_n = pool_n.sample(n_neg, random_state=RANDOM_SEED) if n_neg > 0 else pd.DataFrame()

    lines = ["### FEW-SHOT EXAMPLES (reference only)\n"]
    for _, ex in sampled_p.iterrows():
        d = str(ex[COL_DOC_TEXT])[:200].replace("\n", " ")
        p = str(ex[COL_PAT_TEXT])[:200].replace("\n", " ")
        lines.append(
            f'DEVICE: {d}\nPATENT: {p}\n'
            f'RESULT: {{"reasoning": "Shared specific clinical mechanism and device type.", "linked": "Yes"}}\n---'
        )
    for _, ex in sampled_n.iterrows():
        d = str(ex[COL_DOC_TEXT])[:200].replace("\n", " ")
        p = str(ex[COL_PAT_TEXT])[:200].replace("\n", " ")
        lines.append(
            f'DEVICE: {d}\nPATENT: {p}\n'
            f'RESULT: {{"reasoning": "Despite shared surface terms, clinical domains differ.", "linked": "No"}}\n---'
        )
    lines.append("\n### NOW CLASSIFY THE FOLLOWING (output JSON only):\n")
    return "\n".join(lines)

# ============================================================
# 解析输出
# ============================================================

def parse_response(response) -> int:
    if not response:
        return 0
    s = str(response).lower()
    # 精确匹配优先
    if '"linked": "yes"' in s or '"linked":"yes"' in s:
        return 1
    if '"linked": "no"'  in s or '"linked":"no"'  in s:
        return 0
    # 宽松回退：只看末尾 80 字符，避免 reasoning 里的词干扰
    tail = s[-80:]
    if "yes" in tail and "no" not in tail:
        return 1
    return 0

# ============================================================
# 单轮实验
# ============================================================

def run_experiment(provider, template: str, test_df: pd.DataFrame,
                   pos_pool: pd.DataFrame, neg_pool: pd.DataFrame,
                   k_shot: int) -> dict:
    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                       leave=False, desc=f"{k_shot}-shot"):
        try:
            fda_id       = row.get(COL_FDA_ID, -1)
            entity_score = float(row.get(COL_SCORE_ENTITY, 0))

            few_shot     = build_few_shot_block(pos_pool, neg_pool, k_shot, fda_id)
            core         = fill_prompt(template,
                                       str(row[COL_DOC_TEXT]),
                                       str(row[COL_PAT_TEXT]),
                                       entity_score)
            final_prompt = few_shot + core

            resp = provider._make_api_call(final_prompt)
            pred = parse_response(resp)
            results.append({"label": int(row[COL_LABEL]), "pred": pred})

        except Exception as e:
            logger.warning(f"Sample error: {e}")
            continue

    if not results:
        return {}

    res_df = pd.DataFrame(results)
    tp = int(((res_df.label == 1) & (res_df.pred == 1)).sum())
    fn = int(((res_df.label == 1) & (res_df.pred == 0)).sum())
    fp = int(((res_df.label == 0) & (res_df.pred == 1)).sum())
    tn = int(((res_df.label == 0) & (res_df.pred == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {"recall": recall, "fpr": fpr,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn, "n": len(results)}

# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 65)
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} rows | Labels: {df[COL_LABEL].value_counts().to_dict()}")

    positives = df[df[COL_LABEL] == 1].reset_index(drop=True)
    negatives = df[df[COL_LABEL] == 0].reset_index(drop=True)
    sample_size = min(len(positives), len(negatives), NUM_TEST // 2)

    test_pos = positives.sample(sample_size, random_state=RANDOM_SEED)
    test_neg = negatives.sample(sample_size, random_state=RANDOM_SEED)
    test_df  = pd.concat([test_pos, test_neg]).sample(
        frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # held-out few-shot pool：严格不与 test set 重叠
    fs_pos = positives.drop(index=test_pos.index, errors='ignore').reset_index(drop=True)
    fs_neg = negatives.drop(index=test_neg.index, errors='ignore').reset_index(drop=True)

    print(f"Test set : {len(test_pos)} pos + {len(test_neg)} neg = {len(test_df)} total")
    print(f"Shot pool: {len(fs_pos)} pos, {len(fs_neg)} neg (held-out)")
    print("=" * 65)

    if not LINKING_PROMPT_PATH.exists():
        print(f"Prompt not found: {LINKING_PROMPT_PATH}")
        return
    template = load_template(LINKING_PROMPT_PATH)
    print(f"Prompt loaded: {LINKING_PROMPT_PATH.name}\n")

    
    cfg_sf = LLM_CONFIG.copy()
    cfg_sf["model"] = "deepseek-ai/DeepSeek-V3"

    models = [
        {"label": "GPT-4-turbo", "provider": GPTProvider()},
        {"label": "DeepSeek-V3 (SF)",        "provider": get_provider("sf", worker_id=0, **cfg_sf)},
    ]

    all_results = {}

    for mc in models:
        lbl  = mc["label"]
        prov = mc["provider"]
        print(f"Model: {lbl}")
        all_results[lbl] = {}

        for k in SHOT_LIST:
            print(f"  {k}-shot ... ", end="", flush=True)
            m = run_experiment(prov, template, test_df, fs_pos, fs_neg, k_shot=k)
            all_results[lbl][k] = m
            if m:
                print(f"Recall={m['recall']:.1%}  FPR={m['fpr']:.1%}  "
                      f"(TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']})")
            else:
                print("FAILED")
        print()

    print("=" * 65)
    print("REBUTTAL SUMMARY TABLE")
    print(f"{'Model':<30} {'Shot':>5} | {'Recall':>7} {'FPR':>7}")
    print("-" * 52)
    for lbl, sr in all_results.items():
        for k, m in sr.items():
            if m:
                print(f"{lbl:<30} {k:>4}-shot | {m['recall']:>6.1%} {m['fpr']:>7.1%}")
    print("=" * 65)

    out = DATA_PATH.parent / "rebuttal_fewshot_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
