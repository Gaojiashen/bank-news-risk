"""共享银行别名、分句与银行识别逻辑。"""
import re
from openpyxl import load_workbook
from .paths import ALIASES as ALIAS_FILE
def load_aliases():
    book = load_workbook(ALIAS_FILE, read_only=True, data_only=True)
    try:
        rows = book.active.iter_rows(values_only=True)
        next(rows, None)
        aliases = {}
        for row in rows:
            if len(row) < 2 or not row[1]:
                continue
            bank = str(row[1]).strip()
            aliases[bank] = sorted(
                {str(value).strip() for value in row[1:] if value and len(str(value).strip()) >= 2},
                key=len,
                reverse=True,
            )
        return aliases
    finally:
        book.close()


def split_sentences(text):
    return [part.strip() for part in re.split(r"(?<=[。！？!?；;])|\r?\n+", text) if part.strip()]


def mentioned_banks(sentence, aliases):
    found = []
    for bank, names in aliases.items():
        for name in names:
            if name == "中国银行":
                matched = re.search(r"中国银行(?![间业保])", sentence)
            elif name == "工行":
                matched = re.search(r"(?<!化)工行(?!业)", sentence)
            elif name == "兴业银行":
                matched = re.search(r"(?<!法国)兴业银行", sentence)
            elif name.isascii() and name.isalpha():
                matched = re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", sentence, re.I)
            else:
                matched = name in sentence
            if matched:
                found.append(bank)
                break
    return found
