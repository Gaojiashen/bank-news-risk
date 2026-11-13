"""统一银行名称并解析二分类情绪标签。"""
NAME_ALIASES={"浦发银行":"上海浦东发展银行"}
def label(value: str) -> int | None:
    value = value.strip()
    if value in ("积极", "正面", "1", "+1"):
        return 1
    if value in ("消极", "负面", "-1"):
        return -1
    return None
