# -*- coding: utf-8 -*-
r"""
ESG / GRI 分析邏輯模組 (V12.0 - 支援 Ablation 消融實驗的 Hybrid 架構)
- 結合 BM25 (Sparse) 與 Embedding (Dense) 雙路徑檢索
- 實作 Reciprocal Rank Fusion (RRF) 分數融合
- 支援消融實驗模式切換：full, sparse_only, dense_only, hybrid_no_rule, all (自動跑四次)
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
        "aliases_weak": []
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
        "aliases_weak": []
    },
    "水資源": {
    "primary": ["303-1", "303-2", "303-3", "303-4", "303-5"],
    "related": ["3-3"],

    "aliases_strong": [
        # GRI / 水資源準則
        "GRI 303", "GRI303",
        "GRI 303-1", "GRI 303-2", "GRI 303-3", "GRI 303-4", "GRI 303-5",
        "303-1", "303-2", "303-3", "303-4", "303-5",
        "水與放流水", "Water and Effluents",

        # 國際框架 / 評估工具
        "CDP Water Security", "CDP Water", "Water Security",
        "WRI Aqueduct", "Aqueduct Water Risk Atlas", "Aqueduct",
        "WWF Water Risk Filter", "Water Risk Filter",
        "TNFD", "自然相關財務揭露", "自然資本", "自然資本風險",
        "自然相關風險", "Nature-related risk", "Nature-related",
        "SASB", "ISO 14001", "ISO 46001",

        # 水資源風險 / 水壓力
        "Water Stress", "water stress", "水壓力", "水資源壓力",
        "高水壓力地區", "水壓力地區", "water-stressed area",
        "water stress exposure", "高水壓力地區曝險",
        "缺水風險", "水資源風險", "水風險", "Water Risk",
        "水匱乏", "Water Scarcity", "water scarcity",
        "乾旱", "枯水期", "限水", "供水中斷",
        "洪水", "淹水", "暴雨", "極端降雨",
        "實體風險", "Physical Risk", "氣候實體風險",
        "洪災風險", "乾旱風險",

        # 取水 / 用水 / 耗水
        "取水量", "總取水量", "取水來源", "取水許可",
        "用水量", "總用水量", "耗水量", "水耗用量",
        "Water Withdrawal", "water withdrawal",
        "Water Consumption", "water consumption",
        "Water Use", "water use",
        "Freshwater", "淡水",
        "Other Water", "其他水",
        "第三方水", "Third-party water",
        "地表水", "Surface Water",
        "地下水", "Groundwater",
        "海水", "Seawater",
        "自來水", "市政供水", "Municipal Water",

        # 排水 / 放流水 / 廢水 / 水質
        "排水量", "排放水量", "放流水",
        "廢水", "污水", "製程廢水", "工業廢水",
        "Water Discharge", "water discharge",
        "Effluent", "effluents",
        "廢水處理", "污水處理",
        "廢水處理廠", "污水處理廠",
        "廢水處理設施",
        "放流水標準", "排放水質",
        "水質檢測", "水質監測",
        "放流口", "排放口",
        "水污染防治", "水污染防治措施",
        "排放許可", "許可排放量",
        "COD", "BOD", "SS", "TSS", "pH",
        "氨氮", "總氮", "總磷", "重金屬",
        "化學需氧量", "生化需氧量",

        # 回收 / 再利用 / 循環水
        "回收水", "再生水", "循環水",
        "水回收", "水回收率",
        "回用率", "水再利用率",
        "中水回用", "製程回收水",
        "雨水回收", "再生水使用量",
        "再生水導入",
        "Water Recycling", "Water Reuse",
        "Reclaimed Water", "Recycled Water",
        "Reuse Rate", "Recycling Rate",

        # 水效率 / 水密集度
        "用水效率", "水效率", "water efficiency",
        "水密集度", "用水密集度", "water intensity",
        "單位產品用水量", "單位營收用水量",
        "每單位產量用水",
        "每片晶圓用水量", "單位晶圓用水",
        "per wafer water use", "water use intensity",

        # 半導體製程相關
        "超純水", "UPW", "超純水 UPW",
        "Ultra Pure Water", "Ultrapure Water",
        "製程用水", "清洗用水", "晶圓清洗",
        "純水系統", "DI Water", "去離子水",
        "逆滲透", "RO", "Reverse Osmosis",
        "RO濃排水", "濃排水回收",
        "冷卻水塔", "Cooling Tower",
        "洗滌塔", "Scrubber",
        "冷凝水回收",
        "廢水回收系統", "水回收系統",
        "回收至製程",
        "科學園區供水", "供水穩定",

        # 傳統產業製程相關
        "工業用水", "冷卻用水",
        "循環冷卻水", "鍋爐用水", "洗滌用水",

        # 金融業 / 服務業營運與投融資相關
        "營運據點用水", "辦公室用水", "分行用水",
        "大樓用水", "節水設備", "省水設備",
        "綠建築", "LEED", "EEWH",
        "投融資水風險", "授信水風險",
        "投資組合水風險", "portfolio water risk",
        "水資源投融資風險",
        "水資源敏感產業", "高耗水產業",
        "淡水生態系", "freshwater ecosystem",
        "生物多樣性", "biodiversity"
    ]
}
}


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
# 工具：安全檔名 / 匯出 JSONL+CSV (Excel 可直讀)
# -----------------------------
def _safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    return s[:80] if len(s) > 80 else s


def export_chunks_jsonl_csv(chunks: List[Chunk], company: str, year: str, query: str, ablation_mode: str,
                            out_dir: str = "outputs", max_text_len: int = 4000) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn_company = _safe_filename(company)
    fn_year = _safe_filename(year)
    fn_query = _safe_filename(query)

    # 檔名中加入 mode 方便辨識
    base = f"{fn_company}_{fn_year}_{fn_query}_mode-{ablation_mode}_{stamp}"
    jsonl_path = os.path.join(out_dir, base + ".jsonl")
    csv_path = os.path.join(out_dir, base + ".csv")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            rec = {
                "company": company, "year": year, "topic_query": query, "ablation_mode": ablation_mode,
                "doc_id": ch.doc_id, "page": ch.page, "source": ch.source, "text": ch.text[:max_text_len],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # utf-8-sig 確保可以直接用 Excel 開啟不會亂碼
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        # 完全依照您指定的欄位輸出 (移除了 chunk_id，不包含子面向)
        writer = csv.DictWriter(f, fieldnames=[
            "company", "year", "topic_query", "ablation_mode",
            "doc_id", "page", "source", "char_len", "match_type", "preview"
        ])
        writer.writeheader()
        for i, ch in enumerate(chunks):
            writer.writerow({
                "company": company,
                "year": year,
                "topic_query": query,
                "ablation_mode": ablation_mode,
                "doc_id": ch.doc_id,
                "page": ch.page,
                "source": ch.source,
                "char_len": len(ch.text),
                "match_type": ch.match_type,
                "preview": ch.preview(2000) # 將 preview 長度拉長，確保文本完整
            })

    return jsonl_path, csv_path


# -----------------------------
# 抽取 & OCR
# -----------------------------
def page_text_density(text: str) -> float:
    return sum(1 for c in text if not c.isspace())


def ocr_page_image(page: fitz.Page, dpi: int = 300, lang: str = "chi_tra") -> str:
    if pytesseract is None: return ""
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
    if not line: return False
    return len(line) < 40 and not re.search(r"[。.!?]$", line) and re.search(
        r"(第[一二三四五六七八九十百]+章|\d+\.\d+|目標|政策|管理|章節|附錄)", line)


def adaptive_chunk_v2(pages: List[Chunk], min_chars: int = 350, max_chars: int = 1200, overlap_chars: int = 180,
                      allow_over_max_ratio: float = 1.3) -> List[Chunk]:
    def is_table_like(line: str) -> bool:
        if re.search(r"\d+\s+\d+\s+\d+", line): return True
        if line.count("  ") >= 3: return True
        if "|" in line or "——" in line or "----" in line: return True
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
        if buf: paras.append("\n".join(buf).strip())
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
                    if nxt.startswith("[[p."): break
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
            if acc == 0: return
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
    return bool(re.match(r"^(GRI\s*)?\d{3}(-\d+)?$", s) or re.match(r"^(GRI\s*)?3\s*-\s*3$", s))


def expand_topic_or_gri(q: str) -> Dict[str, Any]:
    q = normalize_query(q)
    tokens = q.split()
    primary, related, aliases_strong, aliases_weak = [], [], [], []
    topic_name = None

    if any(is_gri_code(t) for t in tokens):
        for t in tokens:
            tt = t.upper().replace("GRI", "").strip()
            tt = re.sub(r"\s+", "", tt).replace("–", "-").replace("—", "-")
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
        return {"primary": sorted(set(primary)), "related": sorted(set(related)),
                "aliases_strong": sorted(set(aliases_strong)), "aliases_weak": sorted(set(aliases_weak)),
                "topic_name": topic_name}

    found_topics = []
    for topic, cfg in TOPIC2GRI.items():
        strong, weak = cfg.get("aliases_strong", []), cfg.get("aliases_weak", [])
        if topic in q or any(a in q for a in (strong + weak)):
            primary += cfg.get("primary", [])
            related += cfg.get("related", [])
            aliases_strong += strong
            aliases_weak += weak
            found_topics.append(topic)

    topic_name = " / ".join(found_topics) if found_topics else "一般關鍵字查詢"
    return {"primary": sorted(set(primary)), "related": sorted(set(related)),
            "aliases_strong": sorted(set(aliases_strong)), "aliases_weak": sorted(set(aliases_weak)),
            "topic_name": topic_name}


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
            for t in tokens: tf[t] = tf.get(t, 0) + 1
            self.doc_tfs.append(tf)
            for t in tf.keys(): self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = total_len / len(self.docs) if self.docs else 0.0

    def _score(self, tf: Dict[str, int], q_tokens: List[str], N: int) -> float:
        score = 0.0
        doc_len = sum(tf.values())
        k1, b, avgdl = self.k1, self.b, self.avgdl + 1e-9
        for q in q_tokens:
            doc_freq = self.df.get(q, 0)
            if doc_freq == 0: continue
            idf = math.log(1 + (N - doc_freq + 0.5) / (doc_freq + 0.5))
            term_freq = tf.get(q, 0)
            score += idf * ((term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * doc_len / avgdl)))
        return score

    def search(self, query: str, topk=200) -> List[Tuple[int, float]]:
        q_tokens = self._tokenize(query)
        N = len(self.docs)
        scores = [(i, self._score(tf, q_tokens, N)) for i, tf in enumerate(self.doc_tfs)]
        scores = [s for s in scores if s[1] > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]


# -----------------------------
# RRF 融合演算法
# -----------------------------
def reciprocal_rank_fusion(bm25_hits: List[Tuple[int, float]], dense_hits: List[Tuple[int, float]], k: int = 60) -> \
List[Tuple[int, float]]:
    rrf_scores = {}
    for rank, (idx, _) in enumerate(bm25_hits):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(dense_hits):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    fused = list(rrf_scores.items())
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


# -----------------------------
# 專家摘要 (維持不變)
# -----------------------------
EXPERT_METHOD_HINTS = ["政策", "標準", "程序", "制度", "治理", "KPI", "目標", "監控", "通報", "稽核", "教育訓練",
                       "多層次防護", "帳號管理", "加密", "備援", "災難復原", "自動化", "FTP", "SOC", "DLP", "ISMS",
                       "NICS"]
EXPERT_RISK_HINTS = ["事件", "外洩", "缺失", "違規", "罰款", "風險", "弱點", "異常", "未達", "未見"]


def _collect_metrics_and_signals(chunks: List[Chunk]):
    pass
    return {"numbers": [], "methods": [], "risks": [], "thirdparty": False}


def make_expert_summary(company: str, year: str, query: str, strict_codes: List[str], chunks: List[Chunk]):
    return "自動摘要已略過", "無"


# -----------------------------
# 主流程：混合式分層檢索 (支援 Ablation)
# -----------------------------
def run_pipeline(
        pdf_paths: List[str], company: str, year: str, query: str,
        topk: int = 12, export_all_chunks: bool = True, export_dir: str = "outputs",
        per_page_max: int = 2, density_threshold: int = 200, chunk_min_chars: int = 350,
        chunk_max_chars: int = 1200, chunk_overlap_chars: int = 180,
        ablation_mode: str = "full"
) -> Dict:
    print(f"\n[INFO] 開始處理: {company} {year} | 查詢: {query} | 模式: {ablation_mode}")

    all_pages: List[Chunk] = []
    for p in pdf_paths:
        try:
            all_pages.extend(triage_extract(p, density_threshold=density_threshold))
        except Exception as e:
            print(f"Error processing {p}: {e}")
            continue

    if not all_pages: return {"decision": "錯誤：無法處理任何 PDF 檔案", "selected": []}

    chunks = adaptive_chunk_v2(all_pages, min_chars=chunk_min_chars, max_chars=chunk_max_chars,
                               overlap_chars=chunk_overlap_chars)
    if not chunks: return {"decision": "錯誤：無法從 PDF 中切分段落", "selected": []}

    intents = expand_topic_or_gri(query)
    primary, aliases_strong = intents["primary"], intents.get("aliases_strong", [])

    bm25 = BM25([c.text for c in chunks])
    embedder = get_embedding_model()
    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = embedder.encode(chunk_texts, show_progress_bar=False)

    def contains_code(text: str, code: str) -> bool:
        patt = re.sub(r"\s+", "", code.replace("–", "-").replace("—", "-"))
        return bool(re.search(rf"\b(GRI\s*-?\s*)?{re.escape(patt)}\b", text, re.IGNORECASE))

    ranked_results: List[Tuple[int, float, str]] = []
    processed_indices = set()
    query_is_gri = any(is_gri_code(t) for t in query.split())
    fusion_pool_size = max(topk * 10, 150)
    bm25_hits, dense_hits = [], []

    # ==========================================
    # 核心：消融實驗邏輯
    # ==========================================
    if ablation_mode == "full":
        if query_is_gri or primary:
            direct_codes = primary if query_is_gri else primary
            for i, ch in enumerate(chunks):
                if any(contains_code(ch.text, c) for c in direct_codes):
                    ranked_results.append((i, 9999.0, "Stage 1: Entity Match"))
                    processed_indices.add(i)

    if ablation_mode in ["full", "sparse_only", "hybrid_no_rule"]:
        bm25_query = " ".join(list(dict.fromkeys([query] + aliases_strong)))
        bm25_hits = bm25.search(bm25_query, topk=fusion_pool_size)

    if ablation_mode in ["full", "dense_only", "hybrid_no_rule"]:
        query_embedding = embedder.encode([query])
        sims = cosine_similarity(query_embedding, chunk_embeddings)[0]
        dense_hits = [(i, float(s)) for i, s in enumerate(sims)]
        dense_hits.sort(key=lambda x: x[1], reverse=True)
        dense_hits = dense_hits[:fusion_pool_size]

    fused_results = reciprocal_rank_fusion(bm25_hits, dense_hits)

    for idx, rrf_score in fused_results:
        if idx not in processed_indices:
            in_bm25_top = any(idx == b_idx for b_idx, _ in bm25_hits[:topk]) if bm25_hits else False
            in_dense_top = any(idx == d_idx for d_idx, _ in dense_hits[:topk]) if dense_hits else False

            if ablation_mode == "sparse_only":
                match_label = "Stage 2: Sparse Match (Ablation)"
            elif ablation_mode == "dense_only":
                match_label = "Stage 3: Dense Match (Ablation)"
            else:
                if in_dense_top and not in_bm25_top:
                    match_label = "Stage 3: Dense Match"
                elif in_bm25_top and not in_dense_top:
                    match_label = "Stage 2: Sparse Match"
                else:
                    match_label = "Stage 4: Hybrid Consensus"

            ranked_results.append((idx, float(rrf_score), match_label))
            processed_indices.add(idx)

    ranked_results.sort(key=lambda x: (-x[1], x[0]))

    # --- Top-k 篩選 ---
    selected: List[Chunk] = []
    per_page_count: Dict[int, int] = {}
    seen = set()

    for idx, score, mtype in ranked_results:
        if idx in seen: continue
        ch = chunks[idx]
        page_num = ch.page
        if per_page_count.get(page_num, 0) >= per_page_max: continue

        ch.match_type = mtype
        ch.relevance_score = score
        selected.append(ch)
        per_page_count[page_num] = per_page_count.get(page_num, 0) + 1
        seen.add(idx)
        if len(selected) >= topk: break

    # 4) 匯出 Top-12 的 chunks
    if export_all_chunks:
        jsonl_path, csv_path = export_chunks_jsonl_csv(
            selected, company, year, query, ablation_mode, out_dir=export_dir, max_text_len=4000
        )
        print(f"[EXPORT] CSV 檔案 ({ablation_mode}模式) 已儲存至: {csv_path}")

    return {"selected": selected}


# -----------------------------
# CLI 入口
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", nargs="+", required=True, help="PDF 路徑")
    parser.add_argument("--company", required=True, help="公司名")
    parser.add_argument("--year", required=True, help="年份")
    parser.add_argument("--query", required=True, help="查詢主題")
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--export_dir", default="outputs")
    parser.add_argument("--no_export", action="store_true", help="不輸出檔案")
    parser.add_argument("--per_page_max", type=int, default=2)
    parser.add_argument("--chunk_min_chars", type=int, default=350)
    parser.add_argument("--chunk_max_chars", type=int, default=1200)
    parser.add_argument("--chunk_overlap_chars", type=int, default=180)

    # 包含 "all" 在內的 choices
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "sparse_only", "dense_only", "hybrid_no_rule", "all"],
                        help="選擇消融實驗模式")
    args = parser.parse_args()

    # 當選擇 "all" 時，自動迴圈跑四個模式
    if args.ablation == "all":
        modes_to_run = ["full", "sparse_only", "dense_only", "hybrid_no_rule"]
    else:
        modes_to_run = [args.ablation]

    for mode in modes_to_run:
        print("\n" + "="*50)
        print(f" 開始執行實驗模式: 【 {mode} 】")
        print("="*50)

        res = run_pipeline(
            pdf_paths=args.pdf, company=args.company, year=args.year, query=args.query,
            topk=args.topk, export_all_chunks=not args.no_export, export_dir=args.export_dir,
            per_page_max=args.per_page_max, chunk_min_chars=args.chunk_min_chars,
            chunk_max_chars=args.chunk_max_chars, chunk_overlap_chars=args.chunk_overlap_chars,
            ablation_mode=mode
        )

        print(f"\n==== 終端機預覽 (Top-{args.topk} 證據) [{mode}] ====")
        if "selected" in res and res["selected"]:
            for i, s in enumerate(res["selected"], 1):
                print(f"{i:02d}. p.{s.page} | [{s.match_type}] | {s.preview(100)}")
        else:
            print("未找到任何符合的段落！")

    if args.ablation == "all":
        print("\n 四個消融實驗模式均已執行完畢！檔案已存於 outputs/ 資料夾中。")

if __name__ == "__main__":
    main()