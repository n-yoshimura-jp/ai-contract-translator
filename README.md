# ai-contract-translator

Anthropic Claude API を使って英語⇔日本語の契約書を翻訳するツール群です。
PDF / Word (.docx) / テキスト (.txt) を読み込み、翻訳結果を PDF / Word / テキストの3形式で出力します。

## スクリプト一覧

| スクリプト | 翻訳方向 | 対応入力形式 |
|---|---|---|
| `translate_en2ja.py` | 英語 → 日本語 | PDF / .docx / .txt |
| `translate_ja2en.py` | 日本語 → 英語 | PDF / .docx / .txt |
| `translate_ocr_en2ja.py` | 英語 → 日本語(OCR) | PDF のみ(スキャン画像PDF対応) |
| `translate_ocr_ja2en.py` | 日本語 → 英語(OCR) | PDF のみ(スキャン画像PDF対応) |

`translate_en2ja.py` / `translate_ja2en.py` はテキスト抽出可能なファイル向けです。
`translate_ocr_*.py` はPDFをそのままClaudeのvision機能に送信し、テキスト層のないスキャン画像PDFでも読み取り+翻訳を1回のAPI呼び出しで行います(ページ数・容量に応じて自動でバッチ分割)。

## セットアップ

### 1. 依存ライブラリのインストール

```bash
uv sync
```

または

```bash
pip install -r requirements.txt
```

### 2. APIキーの設定

プロジェクト直下に `.env` ファイルを作成し、以下を記述します。

```
ANTHROPIC_API_KEY=sk-ant-...
```

## 使い方

1. 使いたいスクリプトを開き、先頭の「★★★ 設定 ★★★」ブロックにある `INPUT_FILE` と `OUTPUT_DIR` を書き換える。

   ```python
   INPUT_FILE = "samples/sample_contract_en.pdf"   # 入力ファイルのパス
   OUTPUT_DIR = "output"                            # 出力先ディレクトリ(無ければ自動作成)
   ```

   - 相対パス・絶対パス・`~/` いずれも指定可能です。
   - ドラッグ&ドロップで貼り付けた際の引用符やエスケープ(`\ `)は自動で除去されます。

2. 実行する。

   ```bash
   python translate_en2ja.py
   ```

3. `output/` (デフォルト) に `{元ファイル名}_JA.{txt,docx,pdf}` のような形式で結果が出力されます。

## ディレクトリ構成

```
.
├── translate_en2ja.py       # 英→日翻訳(テキスト抽出版)
├── translate_ja2en.py       # 日→英翻訳(テキスト抽出版)
├── translate_ocr_en2ja.py   # 英→日翻訳(OCR/vision版、PDF専用)
├── translate_ocr_ja2en.py   # 日→英翻訳(OCR/vision版、PDF専用)
├── samples/                 # サンプル契約書(入力例)
├── output/                  # 翻訳結果の出力先(自動生成)
├── requirements.txt
└── pyproject.toml
```

## 注意事項

- OCR版(`translate_ocr_*.py`)は各ページを画像として処理するため、テキスト版よりAPIトークン消費(料金)が多くなります。
- Anthropic APIの制約上、1リクエストあたり最大100ページ/32MBまでのため、OCR版は内部で自動バッチ分割しています。
