#!/usr/bin/env python3
"""
英語契約書 → 日本語翻訳ツール
================================
PDF / Word (.docx) / テキスト (.txt) の英文契約書を読み込み、
Anthropic Claude API で日本語に翻訳し、PDF / Word / テキストの3形式で出力します。

必要ライブラリ:
    pip install anthropic pdfplumber python-docx reportlab python-dotenv

APIキー設定(.env ファイル):
    スクリプトと同じディレクトリに .env ファイルを作成し、以下を記述:
        ANTHROPIC_API_KEY=sk-ant-...
    ※ .env は .gitignore に必ず追加してください(キー漏洩防止)

入出力パスの設定:
    スクリプト先頭の「★★★ 設定 ★★★」ブロックにある
    INPUT_FILE と OUTPUT_DIR を書き換えてください。

使い方:
    1. INPUT_FILE / OUTPUT_DIR を編集
    2. 実行: python translate_en2ja.py
"""

import os
import re
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# .env ファイルから環境変数を読み込む
load_dotenv()

# ===========================================================================
# ★★★ 設定(ここを編集してください)★★★
# ===========================================================================
INPUT_FILE = "samples/sample_contract_en.pdf"   # 入力ファイルのパス(.pdf / .docx / .txt)
OUTPUT_DIR = "output"         # 出力先ディレクトリのパス(無ければ自動作成)
# 例(絶対パスやユーザーホームも指定可能):
#   INPUT_FILE = "/Users/xxx/Documents/service_agreement.docx"
#   INPUT_FILE = "~/Desktop/contract.txt"
#   OUTPUT_DIR = "~/Desktop/翻訳結果"
# ===========================================================================

# ---------------------------------------------------------------------------
# 内部設定
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "claude-sonnet-5"
MAX_CHARS_PER_CHUNK = 8000   # 1回のAPI呼び出しで翻訳する最大文字数(英語)
MAX_TOKENS_PER_CALL = 16384  # 日本語出力は英語よりトークン数が膨らむため多めに確保
MAX_RETRIES = 3              # API失敗時のリトライ回数

SYSTEM_PROMPT = """You are a professional translator specializing in \
English-to-Japanese translation, with expertise across legal contracts, \
business correspondence, technical manuals, academic papers, marketing \
copy, news articles, and general-purpose writing.

Rules:
1. First identify the document's genre and register (legal/contract, \
business, technical, academic, casual, marketing, etc.) from its content \
and tone, and translate into the Japanese style native speakers would \
expect for that genre.
2. For legal or contract documents: use precise, formal Japanese legal \
language (法律文書の文体). Use standard phrasing such as 「〜するものとする」 \
for "shall" and 「〜してはならない」 for "shall not". Preserve article/section \
numbering (Article 1 → 第1条, Section 2.1 → 第2条第1項 or 2.1 as appropriate).
3. For non-legal documents: use natural, idiomatic Japanese appropriate to \
the register (formal business Japanese for business documents, plain but \
precise Japanese for technical/academic text, natural conversational \
Japanese for casual text), while preserving headings, numbered lists, and \
paragraph structure exactly.
4. Use standard terminology for the domain (e.g. contracts: "Party A" → 甲, \
"Party B" → 乙, "this Agreement" → 本契約, "damages" → 損害賠償, \
"confidential information" → 秘密情報, "governing law" → 準拠法) and keep any \
domain-specific or technical terms consistent throughout the document.
5. Translate EVERYTHING faithfully and completely — every sentence, \
heading, list item, table cell, footnote, caption, and label. Do NOT \
summarize, paraphrase away detail, omit, or add anything. Omissions \
(翻訳漏れ) are the most serious possible error and must never happen, even \
for short or repetitive-looking segments.
6. Output ONLY the translated text. No preamble, no commentary, no code fences.
7. If a segment starts or ends mid-sentence, translate it as-is without \
completing the sentence yourself."""


# ---------------------------------------------------------------------------
# 1. ファイル読み込み
# ---------------------------------------------------------------------------
def read_pdf(path: Path) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)

    # テキストが取れなかったページ(スキャン画像ページ)を検出する。
    # 見逃すとそのページが丸ごと翻訳漏れになるため、必ず知らせる。
    blank = [i for i, t in enumerate(pages, 1) if not t.strip()]
    if len(blank) == len(pages):
        raise ValueError(
            "PDFからテキストを抽出できませんでした(全ページが画像)。\n"
            "  スキャン画像のPDFは translate_ocr_en2ja.py を使用してください。"
        )
    if blank:
        pages_str = ", ".join(str(p) for p in blank)
        raise ValueError(
            f"このPDFは一部のページが画像です(テキストを抽出できないページ: "
            f"{pages_str} / 全{len(pages)}ページ)。\n"
            "  このまま翻訳すると該当ページが翻訳漏れになります。\n"
            "  translate_ocr_en2ja.py を使用してください。"
        )
    return "\n\n".join(pages).strip()


def read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    parts = []

    # ヘッダー(本文より前に置く)
    for section in doc.sections:
        parts.extend(_docx_container_text(section.header))

    for para in doc.paragraphs:
        parts.append(para.text)
    # 表(テーブル)内のテキストも取得
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            parts.append(" | ".join(cells))

    # テキストボックス内の文字は doc.paragraphs に含まれないため個別に取得
    parts.extend(_docx_textbox_text(doc))

    # フッター(本文より後に置く)
    for section in doc.sections:
        parts.extend(_docx_container_text(section.footer))

    return "\n".join(parts).strip()


def _docx_container_text(container) -> list[str]:
    """ヘッダー/フッターの段落と表からテキストを取り出す。"""
    parts = [p.text for p in container.paragraphs]
    for table in container.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return [p for p in parts if p.strip()]


def _docx_textbox_text(doc) -> list[str]:
    """テキストボックス(図形)内の文字を取り出す。"""
    from docx.oxml.ns import qn

    parts = []
    for txbx in doc.element.body.iter(qn("w:txbxContent")):
        for para in txbx.iter(qn("w:p")):
            text = "".join(node.text or "" for node in para.iter(qn("w:t")))
            if text.strip():
                parts.append(text)
    return parts


def read_txt(path: Path) -> str:
    # 英文ファイルで一般的なエンコーディングを順に試す
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc).strip()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError("テキストファイルの文字コードを判別できませんでした。")


def read_source(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in (".docx", ".doc"):
        if suffix == ".doc":
            raise ValueError(".doc(旧形式)は未対応です。.docxに変換してください。")
        return read_docx(path)
    if suffix in (".txt", ".text", ".md"):
        return read_txt(path)
    raise ValueError(f"未対応のファイル形式です: {suffix}(PDF / .docx / .txt に対応)")


# ---------------------------------------------------------------------------
# 2. 英語テキストの文字数・単語数カウント
# ---------------------------------------------------------------------------
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")


def count_english_stats(text: str) -> dict[str, int]:
    """英語の単語数と文字数の内訳を返す。"""
    return {
        "英単語数": len(WORD_PATTERN.findall(text)),
        "英字(アルファベット)": len(re.findall(r"[A-Za-z]", text)),
        "数字": len(re.findall(r"[0-9]", text)),
        "総文字数(空白・改行除く)": len(re.sub(r"\s", "", text)),
        "総文字数(全体)": len(text),
    }


def print_char_stats(text: str) -> None:
    counts = count_english_stats(text)
    print("  --- 文字数カウント ---")
    for label, n in counts.items():
        print(f"    {label}: {n:,}")
    print("  ----------------------")


# ---------------------------------------------------------------------------
# 3. 翻訳(Anthropic API)
# ---------------------------------------------------------------------------
# 文末の区切り(句点・ピリオド等と後続の空白)。1文字も失わないよう括弧で捕捉する
SENTENCE_END = re.compile(r"([。．.!?！?]+\s*)")


def split_long_paragraph(para: str, max_chars: int) -> list[str]:
    """改行のない長大な段落を、文の区切りを優先して分割する。

    連結すると元の段落に完全に復元できること(文字を失わないこと)を保証する。
    """
    # 区切り文字も残して分割し、「本文+区切り」を1文として組み立て直す
    tokens = SENTENCE_END.split(para)
    units = []
    for i in range(0, len(tokens), 2):
        unit = tokens[i] + (tokens[i + 1] if i + 1 < len(tokens) else "")
        if unit:
            units.append(unit)

    pieces, current = [], ""
    for unit in units:
        if len(unit) > max_chars and current:
            pieces.append(current)
            current = ""
        while len(unit) > max_chars:      # 1文だけで上限を超える場合は文字数で切る
            pieces.append(unit[:max_chars])
            unit = unit[max_chars:]
        if current and len(current) + len(unit) > max_chars:
            pieces.append(current)
            current = ""
        current += unit
    if current:
        pieces.append(current)
    return pieces or [para]


def split_into_chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """段落境界を保ちながらテキストを分割する(長文契約書対応)。"""
    paragraphs = []
    for para in text.split("\n"):
        # 改行のない長文(1段落が上限超え)はそのままだと訳文が途中で切れるため、
        # 文の区切りを優先して先に分割しておく
        if len(para) > max_chars:
            paragraphs.extend(split_long_paragraph(para, max_chars))
        else:
            paragraphs.append(para)

    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        para_len = len(para) + 1
        if current and current_len + para_len > max_chars:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += para_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def extract_text(response) -> str:
    """レスポンスからテキストブロックだけを取り出す。

    Claude 4.6 以降のモデルは thinking ブロックを返すことがあり、
    content[0] がテキストとは限らないため type で絞り込む。
    """
    return "".join(block.text for block in response.content
                   if block.type == "text")


def translate_chunk(client: anthropic.Anthropic, chunk: str, model: str) -> str:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS_PER_CALL,
                # 翻訳は推論より忠実さが重要。thinking を切って
                # max_tokens を全て訳文に使う(切れによる翻訳漏れを防ぐ)
                thinking={"type": "disabled"},
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (
                        "Translate the following English text "
                        "into Japanese:\n\n" + chunk
                    ),
                }],
            )
            if response.stop_reason == "max_tokens":
                print("  [警告] 出力が max_tokens に達しました。"
                      "訳文が途中で切れている可能性があります"
                      "(MAX_CHARS_PER_CHUNK を小さくしてください)。")
            return extract_text(response).strip()
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
    raise RuntimeError(f"翻訳に失敗しました: {last_error}")


def translate_text(text: str, model: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "エラー: ANTHROPIC_API_KEY が見つかりません。\n"
            "  スクリプトと同じディレクトリに .env ファイルを作成し、\n"
            "  次の1行を記述してください:\n"
            "    ANTHROPIC_API_KEY=sk-ant-..."
        )
    client = anthropic.Anthropic(api_key=api_key)
    chunks = split_into_chunks(text)
    print(f"翻訳を開始します({len(chunks)} チャンク / 約 {len(text):,} 文字)...")

    translated_parts = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  チャンク {i}/{len(chunks)} を翻訳中...")
        translated_parts.append(translate_chunk(client, chunk, model))
    return "\n".join(translated_parts)


# ---------------------------------------------------------------------------
# 4. 出力(TXT / DOCX / PDF)※ 日本語フォント対応
# ---------------------------------------------------------------------------
# 「第1条」「第十二条」などの条見出しを検出するパターン
ARTICLE_PATTERN = re.compile(r"^第[0-9０-９一二三四五六七八九十百]+条")


def write_txt(text: str, path: Path) -> None:
    path.write_text(text + "\n", encoding="utf-8")


def write_docx(text: str, path: Path) -> None:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt

    doc = Document()
    # 日本語フォント設定(欧文フォントと東アジアフォントの両方を指定)
    style = doc.styles["Normal"]
    style.font.name = "Yu Mincho"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "游明朝")

    for line in text.split("\n"):
        line = line.rstrip()
        if not line:
            doc.add_paragraph("")
        elif ARTICLE_PATTERN.match(line):
            p = doc.add_paragraph()
            run = p.add_run(line)
            run.bold = True
        else:
            doc.add_paragraph(line)
    doc.save(str(path))


def write_pdf(text: str, path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    from xml.sax.saxutils import escape

    # 日本語CIDフォントを登録(フォントファイル不要でPDF内蔵の日本語フォントを使用)
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))    # 明朝体(本文)
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))  # ゴシック体(見出し)

    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontName="HeiseiMin-W3", fontSize=10, leading=16,
        wordWrap="CJK",  # 日本語の禁則処理・折り返し
    )
    heading = ParagraphStyle(
        "ArticleHeading", parent=body,
        fontName="HeiseiKakuGo-W5", spaceBefore=8, spaceAfter=4,
    )

    story = []
    for line in text.split("\n"):
        line = line.rstrip()
        if not line:
            story.append(Spacer(1, 6))
        else:
            style = heading if ARTICLE_PATTERN.match(line) else body
            story.append(Paragraph(escape(line), style))
    doc.build(story)


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------
def clean_path(raw: str) -> str:
    """引用符や、Macのドラッグ&ドロップで付くエスケープを取り除く。"""
    s = raw.strip().strip('"').strip("'")
    s = s.replace("\\ ", " ")  # 例: /Users/xx/My\ Docs/contract.pdf
    return os.path.expanduser(s)  # 「~/」をホームディレクトリに展開


def main() -> None:
    # --- コード先頭の設定値からパスを取得 ---
    input_path = Path(clean_path(INPUT_FILE))
    if not input_path.exists():
        sys.exit(
            f"エラー: ファイルが見つかりません: {input_path}\n"
            "  スクリプト先頭の INPUT_FILE を実際のファイルパスに書き換えてください。"
        )

    output_dir = Path(clean_path(OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 読み込み
    print(f"読み込み中: {input_path}")
    source_text = read_source(input_path)
    print("  抽出完了")
    print_char_stats(source_text)

    # 2. 翻訳
    translated = translate_text(source_text, DEFAULT_MODEL)

    # 3. 出力(実行時刻を先頭に付けて、上書きせず時系列に並ぶようにする)
    stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{input_path.stem}_JA"
    outputs = {
        "テキスト": output_dir / f"{stem}.txt",
        "Word":     output_dir / f"{stem}.docx",
        "PDF":      output_dir / f"{stem}.pdf",
    }
    write_txt(translated, outputs["テキスト"])
    write_docx(translated, outputs["Word"])
    write_pdf(translated, outputs["PDF"])

    print("\n完了しました。出力ファイル:")
    for label, p in outputs.items():
        print(f"  [{label}] {p.resolve()}")


if __name__ == "__main__":
    main()