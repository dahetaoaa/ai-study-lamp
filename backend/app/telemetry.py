"""结构化阶段日志，不记录请求头、密钥或学生音频。"""

import json
import logging
from time import perf_counter

log = logging.getLogger("study_lamp")


def elapsed(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)


def timing(stage: str, duration: float, **fields):
    log.info(
        json.dumps(
            {"事件": "阶段耗时", "stage": stage, "duration_ms": duration, **fields}, ensure_ascii=False
        )
    )
