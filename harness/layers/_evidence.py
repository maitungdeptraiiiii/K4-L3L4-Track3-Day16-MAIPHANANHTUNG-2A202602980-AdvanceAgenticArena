"""Bằng chứng dùng chung cho `critic` và `citation_checker`.

Hai lớp đó cùng phải trả lời một câu hỏi: "tài liệu X có thật sự nói câu
này, như một TRÍCH DẪN, và lượt chạy đã truy xuất X chưa?". Module này trả
lời đúng theo cách `arena/scorer.py` chấm:

  * so khớp sau khi chuẩn hoá (NFC + casefold + gộp khoảng trắng) — giống
    `arena.scorer._norm`, nên mô hình thật viết hoa/thường hay thụt lề khác
    đi vẫn được nhận ra. Chuẩn hoá CHỈ dùng để SO SÁNH; không lớp nào ghi
    chữ đã chuẩn hoá ngược vào `claim["text"]`.
  * phạm vi MỘT DÒNG của tài liệu (`arena.scorer._supports`) — câu vắt qua
    hai dòng không phải là trích dẫn.
  * "đã truy xuất" = đã `fetch_doc` hoặc nằm trong kết quả `search`, giống
    tập `retrieved` mà scorer dựng lại từ trace.

Không đọc `Doc.tags`, không hard-code brief hay doc_id.
"""

from __future__ import annotations

import re
import unicodedata

#: Như `arena.scorer.MIN_SUPPORT_CHARS`: ngắn hơn thì không tính là trích dẫn.
MIN_SUPPORT_CHARS = 12

#: Khoá trong `ctx.state` nơi `record_retrieval` ghi các doc_id đã truy xuất.
STATE_KEY = "retrieved_doc_ids"

_WS_RE = re.compile(r"\s+")
_DOC_ID_RE = re.compile(r"doc-\d+")


def norm(text) -> str:
    """Dạng chuẩn hoá mà scorer so sánh trên đó (xem `arena.scorer._norm`)."""
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def _lines(doc) -> list[str]:
    return [line for line in (norm(raw) for raw in doc.body.splitlines()) if line]


def supports(doc, text) -> bool:
    """`doc` chứa `text` nguyên văn trong MỘT dòng (sau chuẩn hoá)."""
    claim = norm(text)
    if doc is None or len(claim) < MIN_SUPPORT_CHARS:
        return False
    return any(claim in line for line in _lines(doc))


def record_retrieval(ctx, name, args, result) -> None:
    """Ghi lại doc_id mà một lượt gọi công cụ vừa truy xuất."""
    seen = ctx.state.setdefault(STATE_KEY, [])
    found: list[str] = []
    if name == "fetch_doc" and result.ok and isinstance(args, dict):
        doc_id = args.get("doc_id")
        if isinstance(doc_id, str):
            found.append(doc_id.strip())
    elif name == "search" and result.ok and isinstance(result.content, str):
        found += _DOC_ID_RE.findall(result.content)
    for doc_id in found:
        if doc_id not in seen:
            seen.append(doc_id)


def retrieved_docs(ctx) -> list:
    """Các Doc lượt chạy đã thật sự nhìn thấy, ưu tiên bản fetch nguyên vẹn.

    Hợp của: doc_id ghi bởi `record_retrieval`, doc_id xuất hiện trong quan
    sát (kết quả search), và tài liệu có body nằm nguyên trong quan sát
    (fetch sạch). Hai nguồn sau giúp lớp vẫn chạy khi đứng một mình.
    """
    if ctx.corpus is None:
        return []
    observed = ctx.observed_text
    ids = set(ctx.state.get(STATE_KEY, [])) | set(_DOC_ID_RE.findall(observed))
    full, partial = [], []
    for doc in ctx.corpus.docs:
        if doc.body and doc.body in observed:
            full.append(doc)
        elif doc.doc_id in ids:
            partial.append(doc)
    return full + partial


def find_source(docs, text):
    """Tài liệu đầu tiên trong `docs` có một dòng chứa `text`, hoặc None."""
    return next((doc for doc in docs if supports(doc, text)), None)


def claim_text(claim) -> str:
    value = claim.get("text") if isinstance(claim, dict) else None
    return value if isinstance(value, str) else ""


def claim_doc_id(claim) -> str:
    """doc_id của claim nếu là chuỗi, ngược lại "" (mô hình thật có thể
    trả list/None — đừng để một kiểu lạ làm `in set` raise)."""
    value = claim.get("doc_id") if isinstance(claim, dict) else None
    return value.strip() if isinstance(value, str) else ""


def sync_citations(report) -> None:
    """`citations` = các doc_id mà claims còn lại thật sự trích."""
    claims = report.get("claims")
    claims = claims if isinstance(claims, list) else []
    report["citations"] = sorted({
        c["doc_id"] for c in claims
        if isinstance(c, dict) and isinstance(c.get("doc_id"), str) and c["doc_id"]
    })
