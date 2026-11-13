"""共享目标银行文本预处理逻辑，保持已选模型的输入规则。"""
import re

class DataError(Exception):
    pass

BANK_ALIASES = {
    "工商银行": ["中国工商银行", "工行"],
    "农业银行": ["中国农业银行", "农行"],
    "中国银行": ["中国银行", "中行"],
    "建设银行": ["中国建设银行", "建行"],
    "交通银行": ["交通银行", "交行"],
    "招商银行": ["招商银行", "招行"],
    "中信银行": ["中信银行", "中信"],
    "光大银行": ["中国光大银行", "光大银行", "光大"],
    "民生银行": ["中国民生银行", "民生银行", "民生"],
    "兴业银行": ["兴业银行", "兴业"],
    "上海浦东发展银行": ["上海浦东发展银行", "浦东发展银行", "浦发银行", "浦发"],
    "平安银行": ["平安银行", "平安"],
    "华夏银行": ["华夏银行", "华夏"],
    "北京银行": ["北京银行"],
    "上海银行": ["上海银行"],
    "江苏银行": ["江苏银行"],
    "南京银行": ["南京银行"],
    "宁波银行": ["宁波银行"],
}


def aliases_for(bank: str) -> list:
    """返回用于定位的候选字符串，长的优先。"""
    names = [bank]
    base = bank.replace("股份有限公司", "").replace("有限公司", "")
    names.append(base)
    names.extend(BANK_ALIASES.get(base, [])) # 没有预设别名时加入空列表
    # list.extend()把可迭代对象的元素逐个加入列表
    if base.startswith("中国") and base.endswith("银行") and len(base) > 4:
        names.append(base[2:])
    seen, out = set(), []
    for n in sorted({n for n in names if n}, key=len, reverse=True):
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def find_mention_span(text: str, bank: str):
    """返回 (start, end) 字符区间；找不到返回 None。长别名优先。"""
    best = None
    for cand in aliases_for(bank):
        pos = text.find(cand)
        if pos < 0:
            continue
        span = (pos, pos + len(cand))
        if best is None or (span[1] - span[0]) > (best[1] - best[0]):
            best = span
    return best


def build_features(rows: list, tokenizer, max_len: int) -> list:
    """
    构造模型输入：
        [CLS] 目标银行 [SEP] 片段（必要时截断） [SEP]
    超长时以“银行被提及位置”为中心取窗口；定位不到时保留首尾各一半并标记。
    两个模型共用这一份构造结果。
    """
    bank_cache = {}
    feats = []
    for r in rows:
        bank = r["bank"]
        if bank not in bank_cache:
            bank_cache[bank] = tokenizer(bank, add_special_tokens=False, # 此时先不加入[CLS][SEP]
                                         verbose=False)["input_ids"] # tokenizer返回的是字典，只取'input_ids'值
        bank_ids = bank_cache[bank]

        enc = tokenizer(r["content"], add_special_tokens=False, verbose=False,
                        return_offsets_mapping=True, truncation=False)
        # return_offsets_mapping返回原文位置，truncation表示截断
        content_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        span = find_mention_span(r["content"], bank)
        mention_tok = None
        if span is not None:
            hit = [i for i, (s, e) in enumerate(offsets) if s < span[1] and e > span[0]]
            # hit表示与目标银行span范围有重叠的token索引，如hit=[1]表示offsets[1]=(0,2)在原文中第1-2字符和目标银行重叠
            if hit:
                mention_tok = hit[0]

        orig_len = len(bank_ids) + len(content_ids) + 3
        budget = max_len - len(bank_ids) - 3 # budget表示content部分最多有多少token
        truncated = len(content_ids) > budget
        mode = "none"
        if truncated:
            if budget <= 0:
                raise DataError(f"银行名过长，无法构造输入：{bank}")
            if mention_tok is not None:
                start = min(max(mention_tok - budget // 2, 0), len(content_ids) - budget)
                content_ids = content_ids[start:start + budget]
                mode = "mention_centered"
            else:
                head = budget // 2
                tail = budget - head
                content_ids = content_ids[:head] + content_ids[-tail:]
                mode = "head_tail_fallback"

        input_ids = [tokenizer.cls_token_id] + bank_ids + [tokenizer.sep_token_id] + \
                    content_ids + [tokenizer.sep_token_id]
        token_type_ids = [0] * (len(bank_ids) + 2) + [1] * (len(content_ids) + 1)

        feats.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "token_type_ids": token_type_ids,
            "truncated": truncated,
            "truncate_mode": mode,
            "mention_found": span is not None,
            "orig_tokens": orig_len,
            "meta": r, # 保留原始数据
        })
    return feats
def mention_spans(text, bank):
    """查找互不重叠的银行提及，优先匹配较长名称而非简称。"""
    spans = []
    for alias in aliases_for(bank):
        for match in re.finditer(re.escape(alias), text):
            span = match.span()
            if not any(span[0] < end and span[1] > start for start, end in spans):
                spans.append(span)
    return sorted(spans)


def build_mention_features(rows, tokenizer, max_len):
    """标记新闻中的目标银行词元，随后汇总全部可见提及。

    token_type_ids用作内部标记：0为银行前缀，1为新闻，
    2为新闻中的目标提及；若提及未保留下来，3表示回退到银行前缀。
    模型在输入BERT前将其还原为0/1句段标识。
    """
    features = []
    for row in rows:
        bank_ids = tokenizer(row["bank"], add_special_tokens=False)["input_ids"]
        encoded = tokenizer(row["content"], add_special_tokens=False,
                            return_offsets_mapping=True, truncation=False)
        content_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        spans = mention_spans(row["content"], row["bank"])
        target_positions = [i for i, (start, end) in enumerate(offsets)
                            if any(start < b and end > a for a, b in spans)]
        budget = max_len - len(bank_ids) - 3
        if budget <= 0:
            raise DataError(f"银行名过长，无法构造输入：{row['bank']}")
        truncated = len(content_ids) > budget
        mode = "none"
        if truncated:
            if target_positions:
                first = target_positions[0]
                start = min(max(first - budget // 2, 0), len(content_ids) - budget)
                indices = list(range(start, start + budget))
                mode = "mention_centered"
            else:
                head = budget // 2
                indices = list(range(head)) + list(range(len(content_ids) - (budget - head), len(content_ids)))
                mode = "head_tail_fallback"
        else:
            indices = list(range(len(content_ids)))
        selected = [content_ids[i] for i in indices]
        target_set = set(target_positions)
        segments = [0] * (len(bank_ids) + 2) + [2 if i in target_set else 1 for i in indices] + [1]
        visible = sum(i in target_set for i in indices)
        if visible == 0:
            # 银行可能仅以未收录的别名或指代表达出现。
            # 保留该样本，并在保存的诊断信息中记录回退。
            segments[1:1 + len(bank_ids)] = [3] * len(bank_ids)
        ids = [tokenizer.cls_token_id] + bank_ids + [tokenizer.sep_token_id] + selected + [tokenizer.sep_token_id]
        features.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                         "token_type_ids": segments, "truncated": truncated,
                         "truncate_mode": mode, "mention_found": bool(target_positions),
                         "mention_count": len(spans), "visible_mention_tokens": visible,
                         "pooling_fallback": visible == 0,
                         "orig_tokens": len(bank_ids) + len(content_ids) + 3,
                         "meta": row})
    return features
