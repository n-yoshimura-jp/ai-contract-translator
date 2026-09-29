"""
翻訳品質チェックツール(誤訳・翻訳漏れの検出)
=====================================================================
原文と翻訳文(PDF / Word / テキスト)を読み込んで比較し、
誤訳・翻訳漏れ・数値の不一致などを検出してレポートを出力します。
原文・翻訳文がスキャン画像PDF(テキスト層なし)の場合は、
Claudeのvision機能で自動的に転写(OCR)してから比較します。

2段階でチェックします:
  1. 機械チェック(Python): 条番号の欠落、数値(金額・日付等)の不一致、
     判読不能マーカーの残存を決定論的に検出
  2. AIレビュー(Claude Opus): 誤訳・翻訳漏れ・不要な追加・
     用語の不統一・法的ニュアンスのずれを詳細にレビュー

日本語⇔英語のどちらの方向でも使用できます(自動判定)。

使い方:
    1. 下の「★★★ 設定 ★★★」ブロックを編集
    2. 実行: python review_translation.py
"""

import base64
import io
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ===========================================================================
# ★★★ 設定(ここを編集してください)★★★
# ===========================================================================
SOURCE_FILE     = "samples/sample_contract_ja.pdf"   # 原文ファイル
TRANSLATED_FILE = "output/20260929_221457_sample_contract_ja_EN.pdf" # 翻訳後ファイル
OUTPUT_DIR      = "output"                           # レポートの出力先

# レビューに使うモデル(重要な契約書は claude-opus-5-5 に切り替えてください)
#   claude-sonnet-5-5 … 通常用。重大な誤訳・翻訳漏れ・数値誤り・無断追加は
#                       十分に検出できる。料金は Opus の1/2
#   claude-opus-5-5   … 重要契約の最終確認用。1語だけの訳し漏れなど軽微な
#                       欠陥まで検出し、実行ごとの結果のブレも小さい
DEFAULT_MODEL = "claude-sonnet-5-5"
# ===========================================================================

# ---------------------------------------------------------------------------
# 内部設定
# ---------------------------------------------------------------------------
TRANSCRIBE_MODEL = "claude-sonnet-5-5"  # スキャンPDFの転写(OCR)用(コスト重視)
TRANSCRIBE_PAGES_PER_BATCH = 10     # 転写時に1回で処理するページ数
MAX_TOKENS_PER_CALL = 32000  # レビューはthinkingが有効で同じ予算を消費するため増量
MAX_RETRIES = 3
MAX_TOTAL_CHARS = 150_000  # 原文+翻訳文の合計がこれを超える場合は警告

SYSTEM_PROMPT = """You are a senior bilingual reviewer specializing in \
Japanese-English translation quality assurance across all document types \
(legal contracts, business correspondence, technical manuals, academic \
papers, casual writing).

Your particular strength is catching defects that survive a casual read: a \
single dropped qualifier, one missing item in an enumeration, a reversed \
direction of obligation, a boilerplate phrase added by the translator. A \
sentence whose overall meaning looks right is not evidence that it is \
correct — you verify it element by element before accepting it."""

REVIEW_PROMPT = """Below are a source document and its translation \
(Japanese-English or English-Japanese; detect the direction yourself).

First identify the document's genre (legal/contract, business, technical, \
academic, casual, marketing, etc.).

# How to review

Work through the SOURCE sentence by sentence, in order. For each sentence, \
find its counterpart in the translation and compare them **element by \
element** — do not stop at "the gist matches". Then do a second pass in the \
reverse direction, reading the TRANSLATION sentence by sentence, to find \
text that has no counterpart in the source.

Verify that each of these survived the translation intact:

- **数値・金額・日付・期間・割合** — 桁、単位、通貨、和暦/西暦
- **固有名詞** — 会社名、氏名、住所、機関名、裁判所名
- **条・項・号の番号と階層** — 番号の対応と、各条の号の**個数**を数えて突合する
- **修飾語・限定語** — "prior", "written", "reasonable", "exclusive", \
"first instance", "material", "sole", "immediately"、「事前の」「書面による」\
「合理的な」「専属的」「第一審の」「重大な」。修飾語が1語落ちるだけで法的効果が \
変わるため、1語単位で照合する
- **並列・列挙の全要素** — "notice or demand"、"A, B, and C" のような並列で \
片方だけが訳されていないか。並列要素の**個数**を原文と訳文で数えて突合する
- **義務・権限の強度と方向** — shall / shall not / may / must と「するものと \
する」「してはならない」「できる」の対応。誰が誰に対して負う義務か、権利が \
どちらからどちらへ移転するか
- **否定・条件・例外** — 「〜を除き」「〜がない限り」"unless", "except", \
"provided, however" の有無と係り先
- **原文にない追加** — 訳文だけにある語句。その言語の契約慣行として自然な \
定型表現(例: 記名押印、署名捺印)であっても、原文に対応表現がなければ「追加」\
として報告する

# What to report

1. 誤訳 (mistranslation): meaning differs from the source
2. 翻訳漏れ (omission): source content missing from the translation \
(sentences, items, dates, names, addresses, headings, table cells, \
footnotes, signature blocks, and **individual words** such as qualifiers)
3. 不要な追加 (addition): content not present in the source
4. 数値・日付の不一致 (number/date mismatch): amounts, dates, article/section \
numbers, periods
5. 用語の不統一 (terminology inconsistency): e.g. 甲/乙 vs Party A/B mapping, \
defined terms or technical terms used inconsistently
6. 法的ニュアンスのずれ (legal nuance — only applicable to legal/contract \
documents): shall/may, 義務/努力義務 の混同など
7. 文体・トーンのずれ (register/tone mismatch): the translation's register \
doesn't match the source document's genre (e.g. casual text translated too \
formally, or vice versa)

# Reporting rules

- **網羅性が最優先。** 重要度で取捨選択してはならない。1語だけの脱落や、 \
実務上は影響が小さいと思われる差異も、気づいたものはすべて報告する。 \
「些細なので省く」という判断は禁止する。severity は「重大/中/軽微」で \
正直に付ければよく、軽微なものを報告しない理由にはならない
- **ただし存在しない問題を作ってはならない。** 各指摘は原文と訳文の該当箇所を \
引用して裏付けられること。裏付けられないものは報告しない
- 以下は問題ではないので報告しない: 表記変換(300,000円 ↔ JPY 300,000、 \
漢数字、序数、和暦/西暦の正しい換算)、語順の自然な入れ替え、その言語として \
自然な言い回しの選択

Respond ONLY with a JSON object in this exact format (no code fences, \
no commentary). Write all descriptions and suggestions in Japanese:

{
  "document_type": "検出した文書の種類(例: 契約書 / ビジネス文書 / 技術文書 / 学術論文 / カジュアルな文章)",
  "overall_assessment": "全体評価を2〜3文で",
  "quality_score": <0-100の整数>,
  "issues": [
    {
      "severity": "重大" | "中" | "軽微",
      "type": "誤訳" | "翻訳漏れ" | "追加" | "数値不一致" | "用語不統一" | "法的ニュアンス" | "文体・トーン",
      "location": "該当箇所(例: 第5条 / Article 5 / 第2段落)",
      "source_excerpt": "原文の該当部分(短く)",
      "translation_excerpt": "翻訳の該当部分(短く。漏れの場合は空文字)",
      "description": "問題の説明",
      "suggestion": "修正案"
    }
  ]
}

If there are no issues, return an empty "issues" array.

=== SOURCE DOCUMENT ===
{SOURCE}

=== TRANSLATION ===
{TRANSLATION}"""


# ---------------------------------------------------------------------------
# 1. ファイル読み込み(PDF / docx / txt)
# ---------------------------------------------------------------------------
def read_source(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n\n".join((p.extract_text() or "") for p in pdf.pages).strip()
        return text  # スキャンPDFの場合は空文字を返す(呼び出し側で転写にフォールバック)
    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        return "\n".join(parts).strip()
    if suffix in (".txt", ".text", ".md"):
        for enc in ("utf-8", "utf-8-sig", "cp932", "shift_jis", "euc-jp",
                    "cp1252", "latin-1"):
            try:
                return path.read_text(encoding=enc).strip()
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"{path.name}: 文字コードを判別できませんでした。")
    raise ValueError(f"未対応のファイル形式です: {suffix}")


def is_japanese(text: str) -> bool:
    jp = len(re.findall(r"[\u3041-\u30FF\u4E00-\u9FFF]", text))
    return jp > len(text) * 0.1


# ---------------------------------------------------------------------------
# 1.5 スキャンPDFの転写(OCR)フォールバック
# ---------------------------------------------------------------------------
TRANSCRIBE_PROMPT = (
    "The attached PDF is a scanned document. Transcribe ALL text exactly as "
    "written, preserving the original language, structure, line breaks, "
    "clause numbering, dates, amounts, names, and addresses. Do NOT "
    "translate, summarize, or correct anything. If a part is illegible, "
    "write [illegible] at that position. Output ONLY the transcription, "
    "with no preamble or commentary."
)


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "エラー: ANTHROPIC_API_KEY が見つかりません。\n"
            "  .env ファイルに ANTHROPIC_API_KEY=sk-ant-... を記述してください。"
        )
    return anthropic.Anthropic(api_key=api_key)


def _pdf_batches(path: Path, pages_per_batch: int) -> list[bytes]:
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(str(path))
    batches = []
    for start in range(0, len(reader.pages), pages_per_batch):
        writer = PdfWriter()
        for page in reader.pages[start: start + pages_per_batch]:
            writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        batches.append(buf.getvalue())
    return batches


def transcribe_scanned_pdf(path: Path, client: anthropic.Anthropic) -> str:
    """テキスト層のないPDFをvision機能で転写する(翻訳はしない)。"""
    batches = _pdf_batches(path, TRANSCRIBE_PAGES_PER_BATCH)
    print(f"  スキャンPDFを転写中({len(batches)} バッチ、"
          f"モデル: {TRANSCRIBE_MODEL})...")
    parts = []
    for i, pdf_bytes in enumerate(batches, 1):
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with client.messages.stream(
                    model=TRANSCRIBE_MODEL,
                    max_tokens=MAX_TOKENS_PER_CALL,
                    # 転写に thinking は不要(トークン節約)。
                    # "between_tools" は Sonnet 5.5 専用のため、モデルを変えたら削除する
                    thinking={"type": "between_tools"},
                    output_config={"effort": "high"},
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "document",
                             "source": {"type": "base64",
                                        "media_type": "application/pdf",
                                        "data": pdf_b64}},
                            {"type": "text", "text": TRANSCRIBE_PROMPT},
                        ],
                    }],
                ) as stream:
                    text = stream.get_final_text()
                    final = stream.get_final_message()
                    if final.stop_reason == "refusal":
                        raise RuntimeError(
                            f"モデルが転写を拒否しました: {final.stop_details}")
                    if final.stop_reason == "max_tokens":
                        print("  [警告] 転写が max_tokens に達しました。"
                              "途中で切れている可能性があります"
                              "(TRANSCRIBE_PAGES_PER_BATCH を小さくしてください)。")
                    parts.append(text.strip())
                break
            except anthropic.AuthenticationError:
                sys.exit(
                    "エラー: APIキーが無効です(401)。"
                    ".env の ANTHROPIC_API_KEY を確認してください。"
                )
            except anthropic.APIError as e:
                last_error = e
                wait = 2 ** attempt
                print(f"  [警告] API エラー(試行 {attempt}/{MAX_RETRIES}): {e}")
                time.sleep(wait)
        else:
            raise RuntimeError(f"転写に失敗しました: {last_error}")
        print(f"  転写バッチ {i}/{len(batches)} 完了")
    return "\n".join(parts)


def load_document(path: Path, client_holder: dict) -> str:
    """ファイルを読み込む。テキスト層のないPDFは自動的に転写する。"""
    text = read_source(path)
    if not text and path.suffix.lower() == ".pdf":
        print(f"  {path.name}: テキスト層なし → visionで転写(OCR)します")
        if client_holder.get("client") is None:
            client_holder["client"] = get_client()
        text = transcribe_scanned_pdf(path, client_holder["client"])
        if not text:
            sys.exit(f"エラー: {path.name} の転写結果が空でした。")
    elif not text:
        sys.exit(f"エラー: {path.name} からテキストを取得できませんでした。")
    return text


# ---------------------------------------------------------------------------
# 2. 機械チェック(決定論的な検証)
# ---------------------------------------------------------------------------
MONTHS = {
    "january": "1", "february": "2", "march": "3", "april": "4",
    "may": "5", "june": "6", "july": "7", "august": "8",
    "september": "9", "october": "10", "november": "11", "december": "12",
}


def extract_numbers(text: str) -> Counter:
    """比較用に数値を正規化して抽出する(全角→半角、月名→数字、カンマ除去)。"""
    t = text.translate(str.maketrans("0123456789", "0123456789"))
    # 月名は直後に数字が続く場合のみ変換する(助動詞 "may" などの誤変換防止)
    for name, num in MONTHS.items():
        t = re.sub(rf"\b{name}\b(?=\s*\d)", f" {num} ", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # 300,000 → 300000
    return Counter(re.findall(r"\d+", t))


def extract_article_numbers(text: str) -> set[int]:
    """条見出し(行頭)のみを対象とする。文中の法令参照
    (例: 著作権法第27条)を誤検出しないよう行頭アンカーで判定する。"""
    if is_japanese(text):
        nums = re.findall(r"^第(\d+)条", text, flags=re.MULTILINE)
    else:
        nums = re.findall(r"^Articles?\s+(\d+)", text,
                          flags=re.MULTILINE | re.IGNORECASE)
    return {int(n) for n in nums}


def mechanical_checks(source: str, translation: str) -> list[str]:
    findings = []

    # (1) 条番号の欠落・過剰
    src_articles = extract_article_numbers(source)
    trans_articles = extract_article_numbers(translation)
    missing = sorted(src_articles - trans_articles)
    extra = sorted(trans_articles - src_articles)
    if missing:
        findings.append(f"[重大] 翻訳に存在しない条番号: {missing}(翻訳漏れの可能性)")
    if extra:
        findings.append(f"[中] 原文に存在しない条番号: {extra}")
    if not missing and not extra and src_articles:
        findings.append(f"[OK] 条番号は一致({len(src_articles)}箇条)")

    # (2) 数値の不一致(月名は数字に正規化して比較)
    src_nums = extract_numbers(source)
    trans_nums = extract_numbers(translation)
    missing_nums = sorted((src_nums - trans_nums).items(),
                          key=lambda x: -len(x[0]))[:15]
    if missing_nums:
        detail = ", ".join(f"{n}(×{c})" for n, c in missing_nums)
        findings.append(
            f"[要確認] 原文にあり翻訳に見つからない数値: {detail}\n"
            "        ※ 表記変換(漢数字・序数など)による誤検出の可能性もあります"
        )
    else:
        findings.append("[OK] 原文の数値はすべて翻訳に存在")

    # (3) 判読不能マーカーの残存(OCR版の出力チェック)
    markers = re.findall(r"\[illegible\]|\[判読不能\]", translation)
    if markers:
        findings.append(f"[中] 判読不能マーカーが {len(markers)} 箇所残っています")

    # (4) 分量バランス(極端に短い=大量の翻訳漏れの疑い)
    ratio = len(translation) / max(len(source), 1)
    findings.append(f"[情報] 文字数比(翻訳/原文): {ratio:.2f} "
                    f"(原文 {len(source):,} 字 → 翻訳 {len(translation):,} 字)")
    if (is_japanese(source) and not is_japanese(translation) and ratio < 1.2) or \
       (not is_japanese(source) and is_japanese(translation) and ratio < 0.25):
        findings.append("[要確認] 翻訳文が原文に対して短すぎます(翻訳漏れの疑い)")

    return findings


# ---------------------------------------------------------------------------
# 3. AIレビュー(Claude Opus)
# ---------------------------------------------------------------------------
def parse_json_response(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(),
                     flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # JSON部分だけを抽出して再試行
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"overall_assessment": "(JSON解析に失敗したため原文を表示)",
            "quality_score": None, "issues": [], "raw": raw}


def ai_review(source: str, translation: str, model: str,
              client: anthropic.Anthropic | None = None) -> dict:
    client = client or get_client()

    user_text = (REVIEW_PROMPT
                 .replace("{SOURCE}", source)
                 .replace("{TRANSLATION}", translation))

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with client.messages.stream(
                model=model,
                max_tokens=MAX_TOKENS_PER_CALL,
                # レビューは判断が必要なため thinking(adaptive)を使う。
                # effort は high 必須。実測(2026-09、誤り8件を仕込んだ訳文)で
                # high は全件検出・誤検出0、medium は重要な漏れを「軽微」に
                # 格下げし誤検出も出た(Opus 5.5 は既定が medium なので明示が必要)
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_text}],
            ) as stream:
                text = stream.get_final_text()
                final = stream.get_final_message()
                if final.stop_reason == "refusal":
                    raise RuntimeError(
                        f"モデルがレビューを拒否しました: {final.stop_details}")
                if final.stop_reason == "max_tokens":
                    print("  [警告] レビュー結果が max_tokens に達しました。"
                          "指摘が途中で切れ、JSON解析に失敗する可能性があります"
                          "(文書を分割してレビューしてください)。")
                return parse_json_response(text)
        except anthropic.AuthenticationError:
            sys.exit(
                "エラー: APIキーが無効です(401)。"
                ".env の ANTHROPIC_API_KEY を確認してください。"
            )
        except anthropic.APIError as e:
            last_error = e
            wait = 2 ** attempt
            print(f"  [警告] API エラー(試行 {attempt}/{MAX_RETRIES}): {e}")
            print(f"  {wait} 秒待機してリトライします...")
            time.sleep(wait)
    raise RuntimeError(f"AIレビューに失敗しました: {last_error}")


# ---------------------------------------------------------------------------
# 4. レポート生成
# ---------------------------------------------------------------------------
SEVERITY_ORDER = {"重大": 0, "中": 1, "軽微": 2}
REPORT_WIDTH = 78   # レポートの折り返し幅(半角換算の桁数)


def _width(text: str) -> int:
    """全角文字を2桁として文字列の表示幅を返す。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WFA" else 1
               for c in text)


def _wrap(text: str, indent: str = "") -> list[str]:
    """表示幅で折り返す。日本語は空白がないため文字単位で折る。"""
    limit = max(20, REPORT_WIDTH - _width(indent))
    out = []
    for para in str(text).split("\n"):
        line, line_w = "", 0
        for ch in para:
            ch_w = _width(ch)
            if line_w + ch_w > limit:
                # 英単語の途中で切らないよう、近くに空白があればそこで折る
                cut = line.rfind(" ")
                if cut > limit // 2:
                    out.append(indent + line[:cut])
                    line = line[cut + 1:]
                else:
                    out.append(indent + line)
                    line = ""
                line_w = _width(line)
            line += ch
            line_w += ch_w
        out.append(indent + line)
    return out


def _field(label: str, value: str, label_w: int = 6) -> list[str]:
    """「ラベル: 値」を、2行目以降もラベル幅だけ字下げして整形する。"""
    pad = " " * max(0, label_w - _width(label))
    head = f"     {label}{pad}: "
    body = _wrap(value, indent=" " * _width(head))
    body[0] = head + body[0].lstrip()
    return body


def build_report(source_path: Path, trans_path: Path,
                 mech: list[str], ai: dict, model: str) -> str:
    bar_eq = "=" * REPORT_WIDTH
    bar_dash = "-" * REPORT_WIDTH
    lines = [bar_eq, " 翻訳品質チェックレポート", bar_eq]
    lines.append(f" 原文          : {source_path}")
    lines.append(f" 翻訳          : {trans_path}")
    lines.append(f" レビューモデル: {model}")
    lines.append("")

    lines += [bar_dash, " 1. 機械チェック(条番号・数値・分量)", bar_dash]
    for f in mech:
        lines += _wrap(f, indent="  ")
    lines.append("")

    lines += [bar_dash, " 2. AIレビュー(誤訳・翻訳漏れ・用語)", bar_dash]
    doc_type = ai.get("document_type")
    if doc_type:
        lines.append(f" 検出した文書の種類: {doc_type}")
    score = ai.get("quality_score")
    if score is not None:
        lines.append(f" 品質スコア        : {score} / 100")

    issues = sorted(ai.get("issues", []),
                    key=lambda x: SEVERITY_ORDER.get(x.get("severity"), 9))
    counts = Counter(i.get("severity", "?") for i in issues)
    lines.append(f" 指摘件数          : {len(issues)} 件"
                 f"(重大 {counts.get('重大', 0)} / 中 {counts.get('中', 0)} / "
                 f"軽微 {counts.get('軽微', 0)})")
    lines.append("")
    lines += _wrap(f"総評: {ai.get('overall_assessment', '')}", indent=" ")
    lines.append("")

    if not issues:
        lines.append(" 指摘事項はありませんでした。")
    else:
        for n, issue in enumerate(issues, 1):
            lines.append(f" [{n}] {issue.get('severity', '?')} / "
                         f"{issue.get('type', '?')} / "
                         f"{issue.get('location', '')}")
            if issue.get("source_excerpt"):
                lines += _field("原文", issue["source_excerpt"])
            if issue.get("translation_excerpt"):
                lines += _field("翻訳", issue["translation_excerpt"])
            lines += _field("問題点", issue.get("description", ""))
            if issue.get("suggestion"):
                lines += _field("修正案", issue["suggestion"])
            lines.append("")

    if "raw" in ai:
        lines += ["", bar_dash, " (参考)AIの生レスポンス", bar_dash]
        lines.append(ai["raw"])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------
def clean_path(raw: str) -> str:
    s = raw.strip().strip('"').strip("'")
    s = s.replace("\\ ", " ")
    return os.path.expanduser(s)


def main() -> None:
    source_path = Path(clean_path(SOURCE_FILE))
    trans_path = Path(clean_path(TRANSLATED_FILE))
    for p, label in ((source_path, "SOURCE_FILE"), (trans_path, "TRANSLATED_FILE")):
        if not p.exists():
            sys.exit(f"エラー: ファイルが見つかりません: {p}\n"
                     f"  スクリプト先頭の {label} を確認してください。")

    output_dir = Path(clean_path(OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    client_holder: dict = {"client": None}
    print(f"原文を読み込み中: {source_path}")
    source = load_document(source_path, client_holder)
    print(f"翻訳を読み込み中: {trans_path}")
    translation = load_document(trans_path, client_holder)

    total = len(source) + len(translation)
    if total > MAX_TOTAL_CHARS:
        print(f"[警告] 合計 {total:,} 文字と長大です。"
              "レビュー精度が落ちる場合は分割を検討してください。")

    print("\n--- 機械チェック ---")
    mech = mechanical_checks(source, translation)
    for f in mech:
        print(f"  {f}")

    print(f"\n--- AIレビュー({DEFAULT_MODEL})を実行中... ---")
    ai = ai_review(source, translation, DEFAULT_MODEL,
                   client=client_holder.get("client"))

    report = build_report(source_path, trans_path, mech, ai, DEFAULT_MODEL)
    # 実行時刻を先頭に付けて、上書きせず時系列に並ぶようにする
    # 翻訳ファイル側の日時プレフィックスは重複するので取り除く
    base = re.sub(r"^\d{8}_\d{6}_", "", trans_path.stem)
    stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{base}_review"
    report_path = output_dir / f"{stem}.txt"
    report_path.write_text(report, encoding="utf-8")

    issues = ai.get("issues", [])
    score = ai.get("quality_score")
    print(f"\nレビュー完了: 指摘 {len(issues)} 件"
          + (f" / 品質スコア {score}/100" if score is not None else ""))
    print(f"レポート: {report_path.resolve()}")


if __name__ == "__main__":
    main()