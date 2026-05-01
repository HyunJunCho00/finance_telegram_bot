# -*- coding: utf-8 -*-

def project_trendline_price(line_info: dict, candle_len: int) -> float | None:
    """Project diagonal trendline value at the latest candle index."""
    if not isinstance(line_info, dict):
        return None
    try:
        x1 = line_info.get("_x1")
        x2 = line_info.get("_x2")
        p1 = line_info.get("point1")
        p2 = line_info.get("point2")
        if x1 is None or x2 is None or not p1 or not p2 or x2 == x1:
            return None
        y1 = float(p1[1])
        y2 = float(p2[1])
        slope = (y2 - y1) / (x2 - x1)
        current_val = y2 + slope * (candle_len - 1 - x2)
        return float(current_val)
    except Exception:
        return None


def pct_change(latest: float | None, prev: float | None) -> float | None:
    if not isinstance(latest, (int, float)) or not isinstance(prev, (int, float)) or prev == 0:
        return None
    return float((latest - prev) / prev * 100.0)


def distance_pct(price: float | None, level: float | None) -> float | None:
    if not isinstance(price, (int, float)) or not isinstance(level, (int, float)) or price == 0:
        return None
    return abs(price - level) / abs(price) * 100.0
