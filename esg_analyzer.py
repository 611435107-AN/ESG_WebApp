# -*- coding: utf-8 -*-
r"""
ESG / GRI 分析邏輯模組 (V11.0 - Hybrid Search 混合檢索架構)
- 結合 BM25 (Sparse) 與 Embedding (Dense) 雙路徑檢索
- 實作 Reciprocal Rank Fusion (RRF) 分數融合
- 四階段平行管線：意圖重構 -> 規則/BM25檢索 -> 向量語意檢索 -> RRF融合與重排
- 新增：關鍵字反查與目錄生成 (Keyword Directory)
- 匯出 chunks 成 JSONL + CSV（供後續 LLM 標註用）

依賴套件：
pip install pymupdf pillow numpy sentence-transformers scikit-learn
（jieba 可選：pip install jieba）
（pytesseract 可選：pip install pytesseract；另需安裝 tesseract 程式）
"""

from __future__ import annotations
import os
import re
import csv
import json
import math
import argparse
import datetime

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import fitz  # PyMuPDF
from PIL import Image
import numpy as np

# 引入 Embedding 與相似度計算套件
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 以下這些選用套件可以用 try，因為不影響核心執行
try:
    import jieba
except ImportError:
    jieba = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

# -----------------------------
# 系統全域變數 (Embedding 模型快取)
# -----------------------------
GLOBAL_EMBEDDING_MODEL = None


def get_embedding_model():
    global GLOBAL_EMBEDDING_MODEL
    if GLOBAL_EMBEDDING_MODEL is None:
        print(">> 正在載入 Embedding 模型 (BAAI/bge-m3)，初次下載可能需要一些時間...")
        # 建議加入 device 參數，確保優先使用 GPU (如有)
        GLOBAL_EMBEDDING_MODEL = SentenceTransformer('BAAI/bge-m3')
    return GLOBAL_EMBEDDING_MODEL


# -----------------------------
# 主題 → GRI 對映
# -----------------------------
TOPIC2GRI = {
    "資安": {
        "primary": ["418-1"],
        "related": ["GRI 2", "3-3"],
        "aliases_strong": [
            "ISMS", "ISO 27001", "ISO/IEC 27001", "ISO 27002", "ISO 27701", "NIST", "SOC 2", "GRI 418-1",
            "SIEM", "DLP", "IAM", "MFA", "多因子驗證", "零信任", "Zero Trust",
            "EDR", "XDR", "WAF", "IDS", "IPS",
            "弱點掃描", "弱掃", "漏洞掃描", "漏洞管理", "弱點管理",
            "滲透測試", "穿透測試", "Penetration Test", "紅隊演練", "Blue Team",
            "加密", "Encryption", "金鑰管理", "KMS", "存取控制", "權限控管",
            "資安事件", "資料外洩", "外洩事件", "事件通報", "通報機制", "事件應變",
            "BCP", "DR", "災難復原", "異地備援", "備援", "營運持續",
            "個資保護", "個人資料", "Personal Data", "隱私", "隱私權",
            "社交工程", "釣魚郵件"
        ],
        "aliases_weak": [
            "資安", "資訊安全", "資通安全", "網路安全", "Cybersecurity",
            "資料安全", "資料保護", "資訊保護",
            "資訊安全管理系統", "資訊安全管理", "資安政策", "資安治理", "資安管理", "CISO",
            "資安風險", "風險評估",
            "資安訓練", "資安教育", "資安教育訓練", "資安意識", "資安演練", "演練", "資安文化", "績效",
            "第三方資安", "供應鏈資安", "委外資安", "雲端安全",
            "SOC", "監控", "log", "日誌", "資安稽核", "資安查核", "稽核",
            "通報"
        ]
    },

    "碳排": {
        "primary": ["305-1", "305-2", "305-3", "305-4", "305-5", "302-1", "302-3", "302-4"],
        "related": ["3-3"],
        "aliases_strong": [
            "tCO2e", "CO2e", "CO2", "GHG", "溫室氣體",
            "Scope 1", "Scope 2", "Scope 3", "範疇一", "範疇二", "範疇三", "範疇一二三",
            "直接排放", "間接排放",
            "GHG Protocol", "溫室氣體盤查議定書", "ISO 14064", "ISO 14067", "IPCC", "CDP", "TCFD",
            "SBTi", "SBT", "RE100", "淨零", "Net Zero", "碳中和", "Carbon Neutral", "淨零 2050",
            "ISO 50001", "GRI302",
            "CCUS", "碳捕捉", "碳封存", "碳移除"
        ],
        "aliases_weak": [
            "碳排", "碳排放", "排放", "排放量", "排放總量", "溫室氣體排放量", "排放總量",
            "碳足跡", "Carbon Footprint",
            "溫室氣體盤查", "碳盤查", "排放盤查", "盤查邊界",
            "減碳", "減排", "減量目標", "減碳目標", "轉型路徑", "減碳路徑", "2030/2050",
            "內部碳定價", "碳定價", "碳費", "碳稅", "碳權", "碳信用", "碳交易",
            "能源效率", "節能", "能效", "用電", "用電量", "電力", "能源使用",
            "再生能源", "綠電", "低碳", "低碳轉型", "製程改善", "設備汰換",
            "減排量", "減碳成效", "相較基準年",
            "GRI", "ISO"
        ]
    },

    "水資源": {
        "primary": ["303-1", "303-2", "303-3", "303-4", "303-5"],
        "related": ["3-3"],
        "aliases_strong": [
            "超純水", "UPW", "超純水 UPW", "製程用水",
            "取水量", "取水來源", "取水許可",
            "地下水", "地表水", "自來水", "地下水/地表水/自來水",
            "回收水", "再生水", "中水回用", "水回收率", "回用率",
            "放流水", "放流標準", "排放水質",
            "WRI Aqueduct", "Water Stress", "水壓力"
        ],
        "aliases_weak": [
            "水資源", "用水", "用水量", "耗水", "耗水量", "取水",
            "排水", "廢水", "污水", "水質",
            "用水管理", "水平衡", "水資源管理", "水管理策略", "節水管理",
            "節水", "省水", "用水效率", "單位用水量", "節水措施",
            "水回收", "循環用水", "水再利用", "減少取水",
            "廢水處理", "污水處理", "處理設施",
            "排放標準", "放流水標準",
            "水風險", "缺水風險", "抗旱", "緊急供水", "供水韌性", "水資源韌性",
            "水足跡", "Water Footprint", "水盤查",
            "冷卻水", "冷卻塔", "雨水回收", "工業用水", "外購水"
        ]
    },
}

TOPIC_OPTIONS = list(TOPIC2GRI.keys())

# -----------------------------
# 資料結構
# -----------------------------
@dataclass
class Chunk:
    doc_id: str
    page: int
    text: str
    source: str
    match_type: str = "未知"
    relevance_score: float = 0.0
    score_meta: Dict[str, Any] = field(default_factory=dict)

    def preview(self, n=200):
        t = self.text.replace("\n", " ")
        return (t[:n] + ("…" if len(t) > n else ""))


# -----------------------------
# 工具：安全檔名 / 匯出 JSONL+CSV
# -----------------------------
def _safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    return s[:80] if len(s) > 80 else s


def export_chunks_jsonl_csv(
        chunks: List[Chunk],
        company: str,
        year: str,
        query: str,
        out_dir: str = "outputs",
        max_text_len: int = 4000
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn_company = _safe_filename(company)
    fn_year = _safe_filename(year)
    fn_query = _safe_filename(query)

    base = f"chunks_{fn_company}_{fn_year}_{fn_query}_{stamp}"
    jsonl_path = os.path.join(out_dir, base + ".jsonl")
    csv_path = os.path.join(out_dir, base + ".csv")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            rec = {
                "chunk_id": f"{fn_company}_{fn_year}_{fn_query}_p{ch.page:03d}_{i:05d}",
                "company": company,
                "year": year,
                "topic_query": query,
                "doc_id": ch.doc_id,
                "page": ch.page,
                "source": ch.source,
                "text": ch.text[:max_text_len],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chunk_id", "company", "year", "topic_query",
                "doc_id", "page", "source",
                "char_len", "preview"
            ]
        )
        writer.writeheader()
        for i, ch in enumerate(chunks):
            chunk_id = f"{fn_company}_{fn_year}_{fn_query}_p{ch.page:03d}_{i:05d}"
            writer.writerow({
                "chunk_id": chunk_id,
                "company": company,
                "year": year,
                "topic_query": query,
                "doc_id": ch.doc_id,
                "page": ch.page,
                "source": ch.source,
                "char_len": len(ch.text),
                "preview": ch.preview(200)
            })

    return jsonl_path, csv_path


# -----------------------------
# 抽取 & OCR
# -----------------------------
def page_text_density(text: str) -> float:
    return sum(1 for c in text if not c.isspace())


def ocr_page_image(page: fitz.Page, dpi: int = 300, lang: str = "chi_tra") -> str:
    if pytesseract is None:
        return ""
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    try:
        txt = pytesseract.image_to_string(img, lang=lang)
    except Exception:
        txt = ""
    return txt


def triage_extract(pdf_path: str, density_threshold: int = 200) -> List[Chunk]:
    doc = fitz.open(pdf_path)
    doc_id = os.path.basename(pdf_path)
    pages: List[Chunk] = []

    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        raw = page.get_text("text") or ""
        dens = page_text_density(raw)

        if dens < density_threshold:
            ocr_txt = ocr_page_image(page)
            text = (raw + "\n" + ocr_txt).strip()
            source = "ocr" if ocr_txt else "pdftext"
        else:
            text = raw
            source = "pdftext"

        if text.strip():
            text = f"[[p.{pno + 1}]]\n" + text
            pages.append(Chunk(doc_id=doc_id, page=pno + 1, text=text, source=source))

    doc.close()
    return pages


# -----------------------------
# 連貫 chunking v2
# -----------------------------
def is_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    return (
            len(line) < 40 and
            not re.search(r"[。.!?]$", line) and
            re.search(r"(第[一二三四五六七八九十百]+章|\d+\.\d+|目標|政策|管理|章節|附錄)", line)
    )


def adaptive_chunk_v2(
        pages: List[Chunk],
        min_chars: int = 350,
        max_chars: int = 1200,
        overlap_chars: int = 180,
        allow_over_max_ratio: float = 1.3
) -> List[Chunk]:
    def is_table_like(line: str) -> bool:
        if re.search(r"\d+\s+\d+\s+\d+", line):
            return True
        if line.count("  ") >= 3:
            return True
        if "|" in line or "——" in line or "----" in line:
            return True
        return False

    def is_bullet_like(line: str) -> bool:
        return bool(re.match(r"^\s*([•\-\*]|(\(?\d+\)?[\.、]))\s+", line))

    def split_paragraphs(text: str) -> List[str]:
        lines = text.splitlines()
        paras = []
        buf = []
        for ln in lines:
            if ln.startswith("[[p."):
                if buf:
                    paras.append("\n".join(buf).strip())
                    buf = []
                paras.append(ln.strip())
                continue

            if not ln.strip():
                if buf:
                    paras.append("\n".join(buf).strip())
                    buf = []
                continue

            buf.append(ln)

        if buf:
            paras.append("\n".join(buf).strip())

        return [p for p in paras if p]

    def merge_table_and_bullets(paras: List[str]) -> List[str]:
        out = []
        i = 0
        while i < len(paras):
            p = paras[i]
            if p.startswith("[[p."):
                out.append(p)
                i += 1
                continue

            lines = p.splitlines()
            tableish = any(is_table_like(ln) for ln in lines)
            bulletish = any(is_bullet_like(ln) for ln in lines)

            if tableish or bulletish:
                j = i + 1
                merged = [p]
                while j < len(paras):
                    nxt = paras[j]
                    if nxt.startswith("[[p."):
                        break
                    nxt_lines = nxt.splitlines()
                    if any(is_table_like(ln) for ln in nxt_lines) or any(is_bullet_like(ln) for ln in nxt_lines):
                        merged.append(nxt)
                        j += 1
                    else:
                        break
                out.append("\n".join(merged).strip())
                i = j
            else:
                out.append(p)
                i += 1
        return out

    out: List[Chunk] = []

    for pg in pages:
        paras = split_paragraphs(pg.text)
        paras = merge_table_and_bullets(paras)

        buf: List[str] = []
        acc = 0
        tail_for_overlap = ""

        def flush():
            nonlocal buf, acc, tail_for_overlap
            if acc == 0:
                return
            text = "\n\n".join(buf).strip()
            if text:
                out.append(Chunk(doc_id=pg.doc_id, page=pg.page, text=text, source=pg.source))
                tail_for_overlap = text[-overlap_chars:] if overlap_chars > 0 else ""
            buf, acc = [], 0

        for p in paras:
            if p.startswith("[[p."):
                flush()
                if tail_for_overlap:
                    buf.append(tail_for_overlap)
                    acc += len(tail_for_overlap)
                buf.append(p)
                acc += len(p)
                continue

            first_line = p.splitlines()[0].strip() if p.splitlines() else ""
            if is_heading(first_line) and acc >= min_chars:
                flush()
                if tail_for_overlap:
                    buf.append(tail_for_overlap)
                    acc += len(tail_for_overlap)

            p_is_table_or_bullet = any(is_table_like(ln) for ln in p.splitlines()) or any(
                is_bullet_like(ln) for ln in p.splitlines())
            limit = int(max_chars * (allow_over_max_ratio if p_is_table_or_bullet else 1.0))

            if acc > 0 and (acc + len(p) > limit) and acc >= min_chars:
                flush()
                if tail_for_overlap:
                    buf.append(tail_for_overlap)
                    acc += len(tail_for_overlap)

            buf.append(p)
            acc += len(p)

            if acc >= max_chars and acc >= min_chars:
                flush()
                if tail_for_overlap:
                    buf.append(tail_for_overlap)
                    acc += len(tail_for_overlap)

        flush()

    return out


# -----------------------------
# 查詢展開
# -----------------------------
def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()


def is_gri_code(s: str) -> bool:
    s = s.strip().upper()
    return bool(
        re.match(r"^(GRI\s*)?\d{3}(-\d+)?$", s) or
        re.match(r"^(GRI\s*)?3\s*-\s*3$", s)
    )


def expand_topic_or_gri(q: str) -> Dict[str, Any]:
    q = normalize_query(q)
    tokens = q.split()
    primary: List[str] = []
    related: List[str] = []
    aliases_strong: List[str] = []
    aliases_weak: List[str] = []
    topic_name: Optional[str] = None

    if any(is_gri_code(t) for t in tokens):
        for t in tokens:
            tt = t.upper().replace("GRI", "").strip()
            tt = re.sub(r"\s+", "", tt)
            tt = tt.replace("–", "-").replace("—", "-")
            if re.match(r"^\d{3}(-\d+)?$", tt) or tt == "3-3":
                primary.append(tt)

        topic_name = "GRI 編號查詢"
        for t_name, cfg in TOPIC2GRI.items():
            if any(code in cfg.get("primary", []) for code in primary):
                related += cfg.get("related", [])
                aliases_strong += cfg.get("aliases_strong", [])
                aliases_weak += cfg.get("aliases_weak", [])
                topic_name = t_name
                break

        return {
            "primary": sorted(set(primary)),
            "related": sorted(set(related)),
            "aliases_strong": sorted(set(aliases_strong)),
            "aliases_weak": sorted(set(aliases_weak)),
            "topic_name": topic_name
        }

    found_topics = []
    for topic, cfg in TOPIC2GRI.items():
        strong = cfg.get("aliases_strong", [])
        weak = cfg.get("aliases_weak", [])
        if topic in q or any(a in q for a in (strong + weak)):
            primary += cfg.get("primary", [])
            related += cfg.get("related", [])
            aliases_strong += strong
            aliases_weak += weak
            found_topics.append(topic)

    if found_topics:
        topic_name = " / ".join(found_topics)
    else:
        topic_name = "一般關鍵字查詢"

    return {
        "primary": sorted(set(primary)),
        "related": sorted(set(related)),
        "aliases_strong": sorted(set(aliases_strong)),
        "aliases_weak": sorted(set(aliases_weak)),
        "topic_name": topic_name
    }


# -----------------------------
# BM25 模型
# -----------------------------
class BM25:
    def __init__(self, docs: List[str], k1=1.5, b=0.75, use_jieba=True):
        self.k1, self.b = k1, b
        self.use_jieba = use_jieba and (jieba is not None)
        self.docs = docs
        self.avgdl = 0.0
        self.df: Dict[str, int] = {}
        self.doc_tfs: List[Dict[str, int]] = []
        self._build()

    def _tokenize(self, s: str) -> List[str]:
        s = s.lower()
        s = re.sub(r"\[\[p\.[^\]]+\]\]", " ", s)
        if self.use_jieba:
            return [t for t in jieba.lcut(s) if t.strip()]
        else:
            return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", s)

    def _build(self):
        total_len = 0
        for doc in self.docs:
            tokens = self._tokenize(doc)
            total_len += len(tokens)
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_tfs.append(tf)
            for t in tf.keys():
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = total_len / len(self.docs) if self.docs else 0.0

    def _score(self, tf: Dict[str, int], q_tokens: List[str], N: int) -> float:
        score = 0.0
        doc_len = sum(tf.values())
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl + 1e-9

        for q in q_tokens:
            doc_freq = self.df.get(q, 0)
            if doc_freq == 0:
                continue
            idf = math.log(1 + (N - doc_freq + 0.5) / (doc_freq + 0.5))
            term_freq = tf.get(q, 0)
            numerator = term_freq * (k1 + 1)
            denominator = term_freq + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * (numerator / denominator)
        return score

    def search(self, query: str, topk=200) -> List[Tuple[int, float]]:
        q_tokens = self._tokenize(query)
        N = len(self.docs)
        scores = []
        for i, tf in enumerate(self.doc_tfs):
            score = self._score(tf, q_tokens, N)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]


# -----------------------------
# Reciprocal Rank Fusion 融合演算法
# -----------------------------
def reciprocal_rank_fusion(bm25_hits: List[Tuple[int, float]], dense_hits: List[Tuple[int, float]], k: int = 60) -> \
List[Tuple[int, float]]:
    """
    結合 Sparse (BM25) 與 Dense (Embedding) 的排名結果。
    輸入皆為 [(chunk_idx, score), ...]，回傳依 RRF 分數降序排列的 [(chunk_idx, rrf_score), ...]
    """
    rrf_scores = {}

    # 處理 BM25 排名
    for rank, (idx, _) in enumerate(bm25_hits):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    # 處理 Dense 排名
    for rank, (idx, _) in enumerate(dense_hits):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    fused_results = list(rrf_scores.items())
    fused_results.sort(key=lambda x: x[1], reverse=True)
    return fused_results


# -----------------------------
# 專家摘要
# -----------------------------
EXPERT_METHOD_HINTS = ["政策", "標準", "程序", "制度", "治理", "KPI", "目標", "監控", "通報", "稽核", "教育訓練",
                       "多層次防護", "帳號管理", "加密", "備援", "災難復原", "自動化", "FTP", "SOC", "DLP", "ISMS",
                       "NICS"]
EXPERT_RISK_HINTS = ["事件", "外洩", "缺失", "違規", "罰款", "風險", "弱點", "異常", "未達", "未見"]


def _collect_metrics_and_signals(chunks: List[Chunk]):
    numbers, methods, risks = [], [], []
    thirdparty = False

    for ch in chunks:
        for ln in (line for line in ch.text.splitlines() if line.strip() and not line.startswith("[[p.")):
            for m in re.finditer(r"\b\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s*(?:%|位|件|筆|項|戶|人|台|萬|kWh|tCO2e)?", ln):
                raw = m.group(0).strip()
                if re.search(r"\d", raw) and not re.match(r"^\d{1,2}$", raw):
                    numbers.append((ch.page, raw))

            low = ln.lower()
            if any(k.lower() in low for k in EXPERT_METHOD_HINTS):
                s = re.sub(r"\s+", " ", ln)
                methods.append((ch.page, s[:160] + ("…" if len(s) > 160 else "")))
            if any(k.lower() in low for k in EXPERT_RISK_HINTS):
                s = re.sub(r"\s+", " ", ln)
                risks.append((ch.page, s[:160] + ("…" if len(s) > 160 else "")))

            if any(k in ln for k in ["第三方", "外部查核", "鑑證", "保證", "ISO", "AA1000"]):
                thirdparty = True

    def uniq(seq):
        seen = set()
        out = []
        for item in seq:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    return {
        "numbers": uniq(numbers)[:10],
        "methods": uniq(methods)[:6],
        "risks": uniq(risks)[:6],
        "thirdparty": thirdparty
    }


def make_expert_summary(company: str, year: str, query: str, strict_codes: List[str], chunks: List[Chunk]):
    pages = sorted(set(ch.page for ch in chunks))
    pages_str = (f"{pages[0]}–{pages[-1]}" if len(pages) > 1 else str(pages[0])) if pages else "?"

    gri_hits_in_selected = set()
    for ch in chunks:
        for code in (strict_codes or []):
            patt = re.sub(r"\s+", "", code.replace("–", "-").replace("—", "-"))
            if re.search(rf"\b(GRI\s*-?\s*)?{re.escape(patt)}\b", ch.text, re.IGNORECASE):
                gri_hits_in_selected.add(code)

    gri_hit_str = "、".join(sorted(gri_hits_in_selected)) if gri_hits_in_selected else "未明列編號"
    sig = _collect_metrics_and_signals(chunks)

    conclusion = [
        f"就「{query}」議題觀察，{company}（{year}）之揭露證據主要集中於 [[p.{pages_str}]]，最終段落與 GRI 對應情形為：{gri_hit_str}。"]
    if sig["numbers"]:
        nice = [n for n in sig["numbers"] if "%" in n[1] or "," in n[1] or "萬" in n[1]] or sig["numbers"]
        topnums = ", ".join(f"[[p.{p}]] {raw}" for p, raw in nice[:3])
        conclusion.append(f"可量化指標可見：{topnums}（擷取原文）。")
    if sig["methods"]:
        conclusion.append("治理/控制作法已見實施（如制度/稽核/監控/教育訓練等），惟仍需持續驗證其成效。")
    if sig["risks"]:
        conclusion.append("同時，文本仍出現風險或事件描述，顯示管控仍在動態精進中。")

    bullets = []
    if sig["methods"]:
        bullets.append("治理與控制")
        for p, s in sig["methods"]:
            bullets.append(f"  • [[p.{p}]] {s}")
    if sig["numbers"]:
        bullets.append("關鍵數據（原文）")
        for p, raw in sig["numbers"]:
            bullets.append(f"  • [[p.{p}]] {raw}")
    if sig["risks"]:
        bullets.append("風險/事件線索")
        for p, s in sig["risks"]:
            bullets.append(f"  • [[p.{p}]] {s}")

    gaps = []
    if not gri_hits_in_selected and strict_codes:
        gaps.append("最終段落未明列 GRI 條款代碼，建議補強與條款對應表或在相關頁面標註。")
    if not sig["thirdparty"]:
        gaps.append("未見第三方查證/保證之描述，可考慮引入外部鑑證或揭露驗證範圍。")
    if not sig["numbers"]:
        gaps.append("缺少可量化 KPI 或年度對比，建議補充年度趨勢與目標達成度。")

    if gaps:
        bullets.append("缺口與建議")
        for g in gaps:
            bullets.append(f"  • {g}")

    return "\n".join(conclusion), ("\n".join(bullets) if bullets else "(尚無可條列之重點)")


# -----------------------------
# 主流程：混合式四階段管線檢索 (Hybrid Search Pipeline)
# -----------------------------
def run_pipeline(
        pdf_paths: List[str],
        company: str,
        year: str,
        query: str,
        topk: int = 12,
        export_all_chunks: bool = True,
        export_dir: str = "outputs",
        per_page_max: int = 2,
        density_threshold: int = 200,
        chunk_min_chars: int = 350,
        chunk_max_chars: int = 1200,
        chunk_overlap_chars: int = 180,
        ablation_mode: str = "full",
) -> Dict:
    # 1) 抽取
    all_pages: List[Chunk] = []
    for p in pdf_paths:
        try:
            all_pages.extend(triage_extract(p, density_threshold=density_threshold))
        except Exception as e:
            print(f"Error processing {p}: {e}")
            continue

    if not all_pages:
        return {
            "query": query, "company": company, "year": year,
            "decision": "錯誤：無法處理任何 PDF 檔案",
            "selected": [],
            "expert_conclusion": "無法處理輸入的 PDF 檔案。",
            "expert_bullets": "",
            "keyword_directory": []
        }

    # 2) 連貫切 chunk
    chunks = adaptive_chunk_v2(
        all_pages,
        min_chars=chunk_min_chars,
        max_chars=chunk_max_chars,
        overlap_chars=chunk_overlap_chars
    )

    if not chunks:
        return {
            "query": query, "company": company, "year": year,
            "decision": "錯誤：無法從 PDF 中切分段落",
            "selected": [],
            "expert_conclusion": "無法從 PDF 文件中有效切分段落。",
            "expert_bullets": "",
            "keyword_directory": []
        }

    # 3) 匯出 chunks
    if export_all_chunks:
        jsonl_path, csv_path = export_chunks_jsonl_csv(
            chunks=chunks, company=company, year=year, query=query, out_dir=export_dir, max_text_len=4000
        )
        print(f"[EXPORT] chunks jsonl -> {jsonl_path}")
        print(f"[EXPORT] chunks csv   -> {csv_path}")

    # ==========================================
    # 第一層 (Stage 1)：意圖重構 (Intent Definition)
    # ==========================================
    intents = expand_topic_or_gri(query)
    primary = intents["primary"]
    related = intents["related"]
    aliases_strong = intents.get("aliases_strong", [])
    aliases_weak = intents.get("aliases_weak", [])  # 加入弱關聯詞以供後續反查標註

    # 載入 BM25 與 Embedding 模型
    bm25 = BM25([c.text for c in chunks])
    embedder = get_embedding_model()

    print(">> 正在將報告書段落進行向量化 (Dense Embedding)...")
    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = embedder.encode(chunk_texts, show_progress_bar=False)

    def contains_code(text: str, code: str) -> bool:
        patt = code.replace("–", "-").replace("—", "-")
        patt = re.sub(r"\s+", "", patt)
        return bool(re.search(rf"\b(GRI\s*-?\s*)?{re.escape(patt)}\b", text, re.IGNORECASE))

    ranked_results: List[Tuple[int, float, str]] = []
    processed_indices = set()
    query_tokens = query.split()
    query_is_gri = any(is_gri_code(t) for t in query_tokens)

    # ==========================================
    # 核心：四階段 Hybrid 檢索邏輯 (支援消融實驗開關)
    # ==========================================
    fusion_pool_size = max(topk * 10, 150)
    bm25_hits = []
    dense_hits = []

    # 【Stage 1 (規則優先)】：GRI 實體精確匹配
    # 消融設定：只有 full 模式才開啟強制規則
    if ablation_mode == "full":
        if query_is_gri or primary:
            direct_codes = primary if query_is_gri else primary
            for i, ch in enumerate(chunks):
                if any(contains_code(ch.text, c) for c in direct_codes):
                    ranked_results.append((i, 9999.0, "Stage 1: Entity Match (實體匹配)"))
                    processed_indices.add(i)

    # 【Stage 2 (Sparse)】：BM25 關鍵字檢索
    # 消融設定：dense_only 模式關閉此層
    if ablation_mode in ["full", "sparse_only", "hybrid_no_rule"]:
        bm25_query = " ".join(list(dict.fromkeys([query] + aliases_strong)))
        bm25_hits = bm25.search(bm25_query, topk=fusion_pool_size)

    # 【Stage 3 (Dense)】：Embedding 語意概念檢索
    # 消融設定：sparse_only 模式關閉此層
    if ablation_mode in ["full", "dense_only", "hybrid_no_rule"]:
        print(f">> 正在執行 Dense 檢索 (Mode: {ablation_mode})...")
        query_embedding = embedder.encode([query])
        sims = cosine_similarity(query_embedding, chunk_embeddings)[0]
        dense_hits = [(i, float(s)) for i, s in enumerate(sims)]
        dense_hits.sort(key=lambda x: x[1], reverse=True)
        dense_hits = dense_hits[:fusion_pool_size]

    # 【Stage 4 (Fusion)】：RRF 分數融合與混合共識重排
    fused_results = reciprocal_rank_fusion(bm25_hits, dense_hits)

    # 併入總榜單，並標記來源引擎
    for idx, rrf_score in fused_results:
        if idx not in processed_indices:
            in_bm25_top = any(idx == b_idx for b_idx, _ in bm25_hits[:topk]) if bm25_hits else False
            in_dense_top = any(idx == d_idx for d_idx, _ in dense_hits[:topk]) if dense_hits else False

            # 根據 ablation_mode 與來源標記 Match Type
            if ablation_mode == "sparse_only":
                match_label = "Stage 2: Sparse Match (Ablation)"
            elif ablation_mode == "dense_only":
                match_label = "Stage 3: Dense Match (Ablation)"
            else:
                if in_dense_top and not in_bm25_top:
                    match_label = "Stage 3: Dense Match (語意概念檢索)"
                elif in_bm25_top and not in_dense_top:
                    match_label = "Stage 2: Sparse Match (關鍵字檢索)"
                else:
                    match_label = "Stage 4: Hybrid Consensus (混合共識)"

            ranked_results.append((idx, float(rrf_score), match_label))
            processed_indices.add(idx)

    # 依最終分數降序排列
    ranked_results.sort(key=lambda x: (-x[1], x[0]))

    # --- Top-k 篩選與每頁上限控制 (Diversity Control) ---
    selected: List[Chunk] = []
    per_page_count: Dict[int, int] = {}
    seen = set()

    for idx, score, mtype in ranked_results:
        if idx in seen:
            continue
        ch = chunks[idx]
        page_num = ch.page

        if per_page_count.get(page_num, 0) >= per_page_max:
            continue

        ch.match_type = mtype
        ch.relevance_score = score

        selected.append(ch)
        per_page_count[page_num] = per_page_count.get(page_num, 0) + 1
        seen.add(idx)

        if len(selected) >= topk:
            break

    strict_codes = primary if primary else [r for r in related if re.match(r"^\d{3}(-\d+)?$|^3-3$", r)]
    expert_conclusion, expert_bullets = make_expert_summary(company, year, query, strict_codes, selected)
    covers_mapped_codes = "是" if bool(strict_codes) else "否"

    # ==========================================
    # [新增] 生成關鍵字目錄 (Keyword Directory)
    # 對檢索出的 Top-K 證據段落進行關鍵字反查
    # ==========================================
    keywords_to_track = set(primary + aliases_strong + aliases_weak + [query])
    keyword_directory_map = {}

    for ch in selected:
        text_lower = ch.text.lower()
        for kw in keywords_to_track:
            # 轉小寫比對確保不漏抓英文縮寫
            if kw.lower() in text_lower:
                if kw not in keyword_directory_map:
                    keyword_directory_map[kw] = set()
                keyword_directory_map[kw].add(ch.page)

    # 整理成列表，依據「出現頁數多寡」由大到小排序，若同分則依字典序排列
    keyword_dir_formatted = []
    for kw, pages in sorted(keyword_directory_map.items(), key=lambda x: (-len(x[1]), x[0])):
        keyword_dir_formatted.append({
            "keyword": kw,
            "pages": sorted(list(pages))
        })
    # ==========================================

    return {
        "query": query,
        "company": company,
        "year": year,
        "decision": "已檢索（Hybrid RRF 架構）",
        "selected": [
            {
                "doc": ch.doc_id,
                "page": ch.page,
                "preview": ch.preview(160),
                "match_type": ch.match_type,
            }
            for ch in selected
        ],
        "expert_conclusion": expert_conclusion,
        "expert_bullets": expert_bullets,
        "covers_mapped_codes": covers_mapped_codes,
        "keyword_directory": keyword_dir_formatted  # 將關鍵字目錄加入回傳結果
    }


# -----------------------------
# CLI 入口
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", nargs="+", required=True, help="PDF 路徑（可多個）")
    parser.add_argument("--company", required=True, help="公司名")
    parser.add_argument("--year", required=True, help="年份，例如 2024")
    parser.add_argument("--query", required=True, help="查詢主題，例如 碳排 / 水資源 / 資安 或 GRI 303-3")
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--export_dir", default="outputs")
    parser.add_argument("--no_export", action="store_true", help="不輸出 chunks 檔案")
    parser.add_argument("--per_page_max", type=int, default=2)
    parser.add_argument("--chunk_min_chars", type=int, default=350)
    parser.add_argument("--chunk_max_chars", type=int, default=1200)
    parser.add_argument("--chunk_overlap_chars", type=int, default=180)
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "sparse_only", "dense_only", "hybrid_no_rule"],
                        help="選擇消融實驗模式")
    args = parser.parse_args()

    res = run_pipeline(
        pdf_paths=args.pdf,
        company=args.company,
        year=args.year,
        query=args.query,
        topk=args.topk,
        export_all_chunks=not args.no_export,
        export_dir=args.export_dir,
        per_page_max=args.per_page_max,
        chunk_min_chars=args.chunk_min_chars,
        chunk_max_chars=args.chunk_max_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
        ablation_mode=args.ablation
    )

    print("\n==== RESULT (selected top-k) ====")
    for i, s in enumerate(res["selected"], 1):
        print(f"{i:02d}. p.{s['page']} | {s['match_type']} | {s['preview']}")

    print("\n==== EXPERT CONCLUSION ====")
    print(res["expert_conclusion"])
    print("\n==== EXPERT BULLETS ====")
    print(res["expert_bullets"])
    print("\n==== covers_mapped_codes ====")
    print(res["covers_mapped_codes"])

    # [新增] 終端機印出關鍵字目錄
    print("\n==== KEYWORD DIRECTORY ====")
    if res.get("keyword_directory"):
        for item in res["keyword_directory"]:
            pages_str = ", ".join(map(str, item["pages"]))
            print(f"- {item['keyword']} (見於 p.{pages_str})")
    else:
        print("(無命中任何預設關鍵字)")


if __name__ == "__main__":
    main()