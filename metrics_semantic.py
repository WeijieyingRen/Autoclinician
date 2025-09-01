# metrics_semantic.py  (embedding-only semantic metrics)
from typing import List, Iterable, Optional, Tuple
import numpy as np
import re
import math

# ---------------------- utils ----------------------
def _basic_norm(s: str) -> str:
    """Very light text normalization (lower, trim, drop bracket meta, collapse spaces)."""
    s = str(s).strip().lower()
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"[()/:,;]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _safe_mean(xs: List[float]) -> float:
    ys = [x for x in xs if isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))]
    return float(sum(ys) / len(ys)) if ys else math.nan


# ---------------- BioEmbedMatcher (bi-encoder) ----------------
class BioEmbedMatcher:
    """
    Embedding-only semantic matcher using sentence-transformers.
    - No lexical exact-match fallback.
    - Cosine similarity via normalized embeddings (inner product).
    """
    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        th_test: float = 0.3,
        th_diag: float = 0.3,
        device: Optional[str] = None,
        cache: bool = True,
    ):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        if device:
            # sentence-transformers 支持 .to(device)；也可在 encode() 里指定 device
            try:
                self.model.to(device)
                self.device = device
            except Exception:
                self.device = None
        else:
            self.device = None
        self.th = {"test": float(th_test), "diag": float(th_diag)}
        self._cache_on = cache
        self._cache = {}

    # ------ embeddings ------
    def _emb(self, texts: List[str]) -> np.ndarray:
        # 简单缓存（注意：长跑可考虑禁用或自行清空以控内存）
        key = (self.model_name, tuple(texts))
        if self._cache_on and key in self._cache:
            return self._cache[key]
        E = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device if self.device else None,
        )
        if self._cache_on:
            self._cache[key] = E
        return E

    # ------ pairwise similarity matrix ------
    def pairwise_scores(self, preds: List[str], golds: List[str]) -> np.ndarray:
        """
        Return cosine similarity matrix S in shape [len(preds), len(golds)].
        If any side is empty, returns an empty matrix with correct dims.
        """
        m, n = len(preds), len(golds)
        if m == 0 or n == 0:
            return np.zeros((m, n), dtype=np.float32)
        P = self._emb([_basic_norm(x) for x in preds])  # [m, d]
        G = self._emb([_basic_norm(x) for x in golds])  # [n, d]
        return P @ G.T                                   # [m, n]

    # ------ single-pred best match ------
    def match_one(self, pred: str, gold_list: Iterable[str], kind: str) -> Optional[int]:
        """
        Compare one pred against all golds; return gold index if max-sim >= threshold else None.
        """
        golds = list(gold_list)
        if not pred or not golds:
            return None
        S = self.pairwise_scores([pred], golds)  # [1, N]
        j = int(np.argmax(S[0])) if S.shape[1] else 0
        return j if (S[0, j] >= self.th[kind]) else None

    # ------ greedy 1:1 over preds (each gold used at most once) ------
    def greedy_set_match(self, preds: List[str], golds: List[str], kind: str) -> Tuple[int, int, int]:
        """
        Iterate preds in order; for each, pick the best still-unmatched gold.
        Count a TP only if max-sim >= threshold. Finally return (tp, fp, fn).
        """
        if not preds and not golds:
            return 0, 0, 0
        if len(golds) == 0:
            return 0, len(preds), 0
        if len(preds) == 0:
            return 0, 0, len(golds)

        S = self.pairwise_scores(preds, golds)  # [M, N]
        used = set()
        tp = 0
        for i in range(S.shape[0]):
            # candidates not yet used
            cand = [(j, S[i, j]) for j in range(S.shape[1]) if j not in used]
            if not cand:
                continue
            j, best = max(cand, key=lambda x: x[1])
            if best >= self.th[kind]:
                tp += 1
                used.add(j)
        fp = len(preds) - tp
        fn = len(golds) - tp
        return tp, fp, fn

    # ------ rowwise best (many-to-one allowed; unique gold counted) ------
    def match_rowwise(self, preds: List[str], golds: List[str], kind: str):
        """
        For each pred (row), take the best gold. A gold can be matched by multiple preds,
        but when counting TP we de-duplicate golds. Return:
          S:     similarity matrix
          pairs: list of (pred_i, best_gold_j, best_score)
          tp, fp, fn: where tp counts unique hit golds, fp counts extra passed preds beyond unique hits.
        """
        S = self.pairwise_scores(preds, golds)
        if S.size == 0:
            return S, [], 0, len(preds), len(golds)

        th = self.th[kind]
        best_j = np.argmax(S, axis=1)                 # [M]
        best_s = S[np.arange(S.shape[0]), best_j]     # [M]
        passed = best_s >= th                         # [M] bool

        pairs = [(int(i), int(best_j[i]), float(best_s[i])) for i in range(len(preds))]
        passed_idx = [i for i in range(len(preds)) if passed[i]]
        hit_golds = set(int(best_j[i]) for i in passed_idx)
        tp = len(hit_golds)
        fp = len(passed_idx) - tp
        fn = len(golds) - tp
        return S, pairs, tp, fp, fn


# ---------------- semantic metrics (greedy by default) ----------------
def f1_semantic(pred: List[str], gold: List[str], kind: str, M: BioEmbedMatcher, mode: str = "greedy") -> float:
    """
    Semantic F1.
    mode:
      - "greedy": 1:1 greedy matching (recommended for set-style evaluation)
      - "rowwise": per-pred best with unique-gold counting
    """
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    if mode == "rowwise":
        _, _, tp, fp, fn = M.match_rowwise(pred, gold, kind)
    else:
        tp, fp, fn = M.greedy_set_match(pred, gold, kind)
    prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    rec  = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

def set_acc_semantic(gold: List[str], pred: List[str], kind: str, M: BioEmbedMatcher) -> float:
    """
    Set-level accuracy (semantic): after greedy 1:1, require fp==0 and fn==0.
    """
    if len(gold) == 0 and len(pred) == 0:
        return 1.0
    tp, fp, fn = M.greedy_set_match(pred, gold, kind)
    return 1.0 if (fp == 0 and fn == 0) else 0.0

def hit_at_k_sem(gold: List[str], pred_ranked: List[str], k: int, kind: str, M: BioEmbedMatcher) -> float:
    """
    语义 Hit@K：Top-K 预测中是否至少有一个与任意 gold 的相似度 >= 阈值。
    """
    if not gold:
        return float("nan")
    if not pred_ranked or k <= 0:
        return 0.0
    Pk = pred_ranked[:k]
    S = M.pairwise_scores(Pk, gold)   # [K, |gold|]
    return 1.0 if (S.size > 0 and float(S.max()) >= M.th[kind]) else 0.0


def recall_at_k_sem(gold: List[str], pred_ranked: List[str], k: int, kind: str, M: BioEmbedMatcher, mode: str = "greedy") -> float:
    """
    Recall@k (semantic). Default greedy.
    - greedy: run 1:1 greedy on top-k preds.
    - rowwise: use rowwise best (unique gold counting) on top-k preds.
    """
    if not gold:
        return math.nan
    k = max(0, int(k))
    Pk = pred_ranked[:k]
    if mode == "rowwise":
        _, _, tp, _, fn = M.match_rowwise(Pk, gold, kind)
    else:
        tp, _, fn = M.greedy_set_match(Pk, gold, kind)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def mrr_sem(gold: List[str], pred_ranked: List[str], kind: str, M: BioEmbedMatcher, k: Optional[int] = None) -> float:
    """
    MRR: position of the first pred that semantically matches ANY gold (>= threshold).
    (Uses greedy-style 'any match' test.)
    """
    if not gold:
        return math.nan
    upto = len(pred_ranked) if k is None else min(k, len(pred_ranked))
    for i in range(upto):
        if M.match_one(pred_ranked[i], gold, kind) is not None:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k_sem(
    gold: List[str],
    pred_ranked: List[str],
    k: int,
    kind: str,
    M: BioEmbedMatcher,
    mode: str = "rowwise"  # "greedy" | "rowwise"
) -> float:
 
    import numpy as np, math
    if not gold: return math.nan
    k = max(0, int(k))
    Pk = pred_ranked[:k]
    if not Pk: return 0.0

    gains = []
    used = set()
    th = M.th[kind]

    if mode == "rowwise":
        # 先算相似度矩阵，逐行取最佳；同一 gold 只在首次出现时计 1
        S = M.pairwise_scores(Pk, gold)  # [m, n]
        best_j = np.argmax(S, axis=1) if S.size else np.array([], dtype=int)
        best_s = S[np.arange(S.shape[0]), best_j] if S.size else np.array([], dtype=float)
        for i in range(len(Pk)):
            j = int(best_j[i]) if i < len(best_j) else -1
            s = float(best_s[i]) if i < len(best_s) else 0.0
            if j >= 0 and (j not in used) and s >= th:
                gains.append(1.0)
                used.add(j)
            else:
                gains.append(0.0)

    else:  # greedy
        # 按顺序对“未用过的 gold”做匹配（与你现有实现一致）
        remain_idx = [j for j in range(len(gold)) if j not in used]
        for i in range(len(Pk)):
            # 只在未使用 gold 上找最佳
            remain = [gold[j] for j in remain_idx]
            idx = M.match_one(Pk[i], remain, kind)
            if idx is not None:
                real = remain_idx[idx]
                used.add(real)
                gains.append(1.0)
                # 更新剩余 gold 索引
                remain_idx = [j for j in range(len(gold)) if j not in used]
            else:
                gains.append(0.0)

    # DCG / IDCG（二值增益）
    dcg  = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    m    = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(m))
    return 0.0 if idcg == 0 else float(dcg / idcg)


# ---------------- dataset aggregations ----------------
def micro_f1_semantic(all_preds: List[List[str]], all_golds: List[List[str]], kind: str, M: BioEmbedMatcher, mode: str = "greedy") -> float:
    """
    Micro-F1 across dataset.
    """
    TP = FP = FN = 0
    for preds, golds in zip(all_preds, all_golds):
        if mode == "rowwise":
            _, _, tp, fp, fn = M.match_rowwise(preds, golds, kind)
        else:
            tp, fp, fn = M.greedy_set_match(preds, golds, kind)
        TP += tp; FP += fp; FN += fn
    denom = 2 * TP + FP + FN
    return 0.0 if denom == 0 else (2 * TP) / denom

def macro_f1_semantic(per_turn_f1s: List[float]) -> float:
    """Mean over turns (ignore NaN/Inf)."""
    return _safe_mean(per_turn_f1s)
