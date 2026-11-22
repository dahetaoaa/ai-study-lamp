"""知识图谱聚合、学生当前进度投影与会话实时批改融合服务。"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _load_json(name: str) -> dict[str, Any]:
    path = HERE / name
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# 将试卷题目的考点 tag 映射到知识图谱节点，实现批改结果实时联动知识图谱掌握度！
MATH_TAG_RULES: list[tuple[tuple[str, ...], str]] = [
    (("一元二次方程的定义", "一元二次方程的根"), "21.1"),
    (("直接开平方法", "配方法"), "21.2.1"),
    (("判别式", "公式法"), "21.2.2"),
    (("因式分解法", "一元二次方程的解法"), "21.2.3"),
    (("韦达定理", "根与系数的关系"), "21.2.4"),
    (("一元二次方程的实际应用", "传播问题", "面积问题"), "21.3"),
    (("二次函数图象", "二次函数的图象", "顶点式", "对称性", "增减性", "开口大小", "平移", "待定系数法", "最值"), "22.1"),
    (("二次函数与坐标轴", "二次函数与一次函数", "抛物线与x轴"), "22.2"),
    (("二次函数实际应用", "最大利润", "二次函数综合"), "22.3"),
    (("中位数", "众数", "平均数"), "20.1"),
    (("方差", "波动"), "20.2"),
    (("旋转", "中心对称"), "23.1"),
    (("圆", "垂径", "圆周角", "切线"), "24.1"),
    (("概率", "随机事件"), "25.1"),
    (("反比例函数",), "26.1"),
    (("相似", "位似"), "27.2"),
    (("三角函数", "解直角三角形"), "28.1"),
]

EN_TAG_RULES: list[tuple[tuple[str, ...], str]] = [
    (("by + doing", "by hand", "询问学习", "学习方法"), "en_9_u1"),
    (("感叹句", "宾语从句引导词", "if 引导宾语从句", "节日"), "en_9_u2"),
    (("礼貌问路", "宾语从句语序", "疑问词+不定式", "校园场所"), "en_9_u3"),
    (("used to",), "en_9_u4"),
    (("被动语态", "一般现在时的被动", "传统非遗", "文化介绍"), "en_9_u5"),
    (("一般过去时的被动",), "en_9_u6"),
    (("情态动词",), "en_9_u7"),
    (("定语从句",), "en_9_u9"),
    (("现在完成", "过去分词"), "en_8b_u8"),
    (("条件状语从句", "一般将来时"), "en_8a_u10"),
    (("原因状语从句", "让步状语从句", "时间状语从句"), "en_8b_u6"),
    (("最高级", "比较级"), "en_8a_u4"),
    (("反身代词",), "en_8b_u1"),
    (("不定代词",), "en_8a_u1"),
    (("there be", "方位介词"), "en_7b_u8"),
    (("动名词", "动词不定式"), "en_8b_u2"),
]


def _det_mastery(seed_key: str, low: int, high: int) -> int:
    h = 2166136261
    for ch in seed_key:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return low + (h % (high - low + 1))


@lru_cache(maxsize=1)
def load_base_graphs() -> dict[str, dict[str, Any]]:
    g7 = _load_json("raw_math_g7.json").get("books", [])
    g89 = _load_json("raw_math_g8_g9.json").get("books", [])
    math_books = [*g7, *g89]

    en78 = _load_json("raw_en_g7_g8.json").get("books", [])
    en9_raw = _load_json("raw_en_g9.json")
    en9 = en9_raw.get("books", [])
    grammar_clusters = en9_raw.get("grammar_clusters", [])
    en_books = [*en78, *en9]

    return {
        "math": _build_math_graph(math_books),
        "en": _build_en_graph(en_books, grammar_clusters),
    }


def _build_math_graph(books: list[dict[str, Any]]) -> dict[str, Any]:
    # 当前学习锚点：九年级上册 · 第二十一章 一元二次方程 · 21.2.3 因式分解法
    current_code = "21.2.3"
    all_codes: list[str] = []
    for b in books:
        for ch in b.get("chapters", []):
            for sec in ch.get("sections", []):
                all_codes.append(sec["code"])

    cur_idx = all_codes.index(current_code) if current_code in all_codes else max(0, len(all_codes) - 1)
    code_set = set(all_codes)
    edges: list[dict[str, str]] = []
    lit_count = 0
    total_count = len(all_codes)

    enriched_books = copy.deepcopy(books)
    seq = 0
    for b in enriched_books:
        b_lit = 0
        b_total = 0
        for ch in b.get("chapters", []):
            ch_lit = 0
            ch_mastery_sum = 0
            secs = ch.get("sections", [])
            for i, sec in enumerate(secs):
                code = sec["code"]
                sec["node_id"] = f"math_{code}"
                sec["order"] = seq
                ew = sec.get("exam_weight")
                if isinstance(ew, (int, float)):
                    sec["exam_weight"] = "压轴常考" if ew >= 0.9 else "高频核心" if ew >= 0.75 else "基础必考"
                elif not ew:
                    sec["exam_weight"] = "核心考点"
                if seq < cur_idx:
                    sec["status"] = "mastered"
                    base_m = _det_mastery(code, 78, 97)
                    # 第21章刚学完的前几节掌握度贴合 74% 章节画像
                    if ch.get("num") == 21:
                        base_m = {"21.1": 86, "21.2": 78, "21.2.1": 80, "21.2.2": 74}.get(code, 78)
                    sec["mastery"] = base_m
                    sec["today"] = ch.get("num") == 21 and code in ("21.2.2", "21.2.3")
                    lit_count += 1
                    b_lit += 1
                    ch_lit += 1
                elif seq == cur_idx:
                    sec["status"] = "learning"
                    sec["mastery"] = 68
                    sec["today"] = True
                    lit_count += 1
                    b_lit += 1
                    ch_lit += 1
                elif ch.get("num") == 21:
                    sec["status"] = "next"
                    sec["mastery"] = 25 if i == secs.index(sec) else 0
                    sec["today"] = False
                else:
                    sec["status"] = "locked"
                    sec["mastery"] = 0
                    sec["today"] = False

                ch_mastery_sum += sec["mastery"]
                b_total += 1
                seq += 1

                prereqs = [p for p in sec.get("prereqs", []) if p in code_set]
                if not prereqs and i > 0:
                    prereqs = [secs[i - 1]["code"]]
                sec["prereqs"] = prereqs
                for p in prereqs:
                    edges.append({"from": p, "to": code})

            ch["total_sections"] = len(secs)
            ch["lit_sections"] = ch_lit
            if ch.get("num") == 21:
                ch["status"] = "learning"
                ch["mastery"] = 74
            elif ch_lit == len(secs) and secs:
                ch["status"] = "mastered"
                ch["mastery"] = round(ch_mastery_sum / len(secs))
            else:
                ch["status"] = "locked"
                ch["mastery"] = 0
        b["total_sections"] = b_total
        b["lit_sections"] = b_lit
        b["progress_pct"] = round(b_lit * 100 / b_total) if b_total else 0

    pct = round(lit_count * 100 / total_count) if total_count else 70
    return {
        "subject": "math",
        "title": "初中数学",
        "en_title": "Mathematics",
        "publisher": "人教版 · 七年级上册 — 九年级下册",
        "current": {
            "book_id": "math_9a",
            "book_title": "九年级上册",
            "chapter_id": "math_ch21",
            "chapter_no": "第二十一章",
            "chapter_title": "一元二次方程",
            "section_code": current_code,
            "section_title": "因式分解法",
            "chapter_mastery": 74,
            "today_focus": "提公因式与十字相乘法解方程 · 警惕两边同除含未知数因式导致丢根",
        },
        "stats": {
            "lit_nodes": lit_count,
            "total_nodes": total_count,
            "progress_pct": pct,
            "books_count": len(enriched_books),
            "chapters_count": sum(len(b.get("chapters", [])) for b in enriched_books),
            "domains": ["数与代数", "图形与几何", "统计与概率"],
        },
        "books": enriched_books,
        "edges": edges,
    }


def _build_en_graph(books: list[dict[str, Any]], grammar_clusters: list[dict[str, Any]]) -> dict[str, Any]:
    # 当前学习锚点：九年级全一册 · Unit 5 What are the shirts made of?
    current_uid = "en_9_u5"
    all_uids: list[str] = []
    for b in books:
        for u in b.get("units", []):
            all_uids.append(u["unit_id"])

    cur_idx = all_uids.index(current_uid) if current_uid in all_uids else max(0, len(all_uids) - 1)
    uid_set = set(all_uids)
    edges: list[dict[str, str]] = []
    lit_units = 0
    total_units = len(all_uids)
    lit_vocab = 0
    total_vocab = 0

    enriched_books = copy.deepcopy(books)
    seq = 0
    prev_uid = None
    for b in enriched_books:
        b_lit = 0
        units = b.get("units", [])
        for u in units:
            uid = u["unit_id"]
            v_cnt = int(u.get("vocab_count", 28))
            total_vocab += v_cnt
            u["order"] = seq
            if seq < cur_idx:
                u["status"] = "mastered"
                u["mastery"] = _det_mastery(uid, 80, 96)
                u["today"] = uid == "en_9_u4"
                lit_units += 1
                b_lit += 1
                lit_vocab += v_cnt
            elif seq == cur_idx:
                u["status"] = "learning"
                u["mastery"] = 76
                u["today"] = True
                lit_units += 1
                b_lit += 1
                lit_vocab += round(v_cnt * 0.68)
            elif b.get("book_id") == "en_9" and seq <= cur_idx + 2:
                u["status"] = "next"
                u["mastery"] = 20 if seq == cur_idx + 1 else 0
                u["today"] = False
            else:
                u["status"] = "locked"
                u["mastery"] = 0
                u["today"] = False

            prereqs = [p for p in u.get("prereqs", []) if p in uid_set and p != uid]
            if not prereqs and prev_uid:
                prereqs = [prev_uid]
            u["prereqs"] = prereqs
            for p in prereqs:
                edges.append({"from": p, "to": uid})

            prev_uid = uid
            seq += 1

        b["total_units"] = len(units)
        b["lit_units"] = b_lit
        b["progress_pct"] = round(b_lit * 100 / len(units)) if units else 0

    # 构建单元索引供语法专题集群计算掌握度
    unit_map = {u["unit_id"]: u for b in enriched_books for u in b.get("units", [])}
    enriched_clusters = copy.deepcopy(grammar_clusters)
    for gc in enriched_clusters:
        uids = [x for x in gc.get("unit_ids", []) if x in unit_map]
        gc["unit_ids"] = uids
        if uids:
            lit_in = [unit_map[x] for x in uids if unit_map[x]["status"] in ("mastered", "learning")]
            gc["lit_count"] = len(lit_in)
            gc["total_count"] = len(uids)
            gc["mastery"] = round(sum(unit_map[x]["mastery"] for x in uids) / len(uids))
            gc["active"] = current_uid in uids
        else:
            gc["lit_count"] = 0
            gc["total_count"] = 0
            gc["mastery"] = 0
            gc["active"] = False

    pct = round(lit_units * 100 / total_units) if total_units else 80
    return {
        "subject": "en",
        "title": "初中英语",
        "en_title": "English",
        "publisher": "人教版 Go for it! · 七年级上册 — 九年级全一册",
        "current": {
            "book_id": "en_9",
            "book_title": "九年级全一册",
            "unit_id": current_uid,
            "unit_code": "Unit 5",
            "unit_title": "What are the shirts made of?",
            "grammar_focus": "一般现在时的被动语态 · 主将从现句法衔接",
            "featured_word": {
                "word": "glass",
                "phonetic": "/ɡlɑːs/",
                "pos": "n.",
                "meaning": "玻璃；玻璃杯",
                "phrase": "made of glass",
                "source": "九全 Unit 5",
            },
            "unit_mastery": 76,
            "today_focus": "be made of / from / in 辨析 · 一般现在时被动语态 am/is/are + done",
        },
        "stats": {
            "lit_nodes": lit_units,
            "total_nodes": total_units,
            "progress_pct": pct,
            "vocab_learned": lit_vocab,
            "vocab_total": total_vocab,
            "books_count": len(enriched_books),
        },
        "books": enriched_books,
        "grammar_clusters": enriched_clusters,
        "edges": edges,
    }


def match_question_node(subject: str, tag: str | None) -> str | None:
    if not tag:
        return None
    rules = MATH_TAG_RULES if subject == "math" else EN_TAG_RULES
    for keywords, node_key in rules:
        if any(k in tag for k in keywords):
            return node_key
    return None


def get_subject_graph(subject: str, session_data: dict[str, Any] | None = None) -> dict[str, Any]:
    graphs = load_base_graphs()
    if subject not in graphs:
        raise KeyError(subject)
    graph = copy.deepcopy(graphs[subject])
    if not session_data:
        return graph

    # 融合当前会话实时批改题目，动态反映“今天刚做过哪些题、对错如何”
    captures = session_data.get("captures", {})
    questions = session_data.get("questions", {})
    node_records: dict[str, dict[str, Any]] = {}
    for q in questions.values():
        cap = captures.get(q.get("capture_id"), {})
        q_sub = cap.get("subject")
        if q_sub != subject:
            continue
        node_key = match_question_node(subject, q.get("tag"))
        if not node_key:
            continue
        rec = node_records.setdefault(node_key, {"total": 0, "right": 0, "wrong": 0, "questions": []})
        status = q.get("status")
        if status in ("right", "wrong", "blank"):
            rec["total"] += 1
            if status == "right":
                rec["right"] += 1
            elif status == "wrong":
                rec["wrong"] += 1
            rec["questions"].append(
                {
                    "question_uid": q.get("question_uid"),
                    "no": q.get("no"),
                    "status": status,
                    "tag": q.get("tag"),
                }
            )

    if subject == "math" and node_records:
        for b in graph["books"]:
            for ch in b["chapters"]:
                for sec in ch["sections"]:
                    rec = node_records.get(sec["code"])
                    if rec:
                        sec["session_practice"] = rec
                        sec["today"] = True
                        if sec["status"] == "locked":
                            sec["status"] = "learning"
                        if rec["total"]:
                            acc = round(rec["right"] * 100 / rec["total"])
                            sec["mastery"] = round(sec["mastery"] * 0.6 + acc * 0.4) if sec["mastery"] else acc
    elif subject == "en" and node_records:
        for b in graph["books"]:
            for u in b["units"]:
                rec = node_records.get(u["unit_id"])
                if rec:
                    u["session_practice"] = rec
                    u["today"] = True
                    if u["status"] == "locked":
                        u["status"] = "learning"
                    if rec["total"]:
                        acc = round(rec["right"] * 100 / rec["total"])
                        u["mastery"] = round(u["mastery"] * 0.6 + acc * 0.4) if u["mastery"] else acc

    return graph


def get_overview_payload(session_data: dict[str, Any] | None = None) -> dict[str, Any]:
    math_g = get_subject_graph("math", session_data)
    en_g = get_subject_graph("en", session_data)

    questions = list((session_data or {}).get("questions", {}).values())
    scored_qs = [q for q in questions if q.get("qtype") not in ("listening", "writing")]
    done_qs = [q for q in scored_qs if q.get("status") in ("right", "wrong", "blank")]
    wrong_qs = [q for q in scored_qs if q.get("status") == "wrong"]
    caps = list((session_data or {}).get("captures", {}).values())

    math_lit = math_g["stats"]["lit_nodes"]
    math_tot = math_g["stats"]["total_nodes"]
    en_lit = en_g["stats"]["lit_nodes"]
    en_tot = en_g["stats"]["total_nodes"]
    overall_lit_pct = (
        round((math_lit + en_lit) * 100 / (math_tot + en_tot)) if (math_tot + en_tot) else 74
    )

    return {
        "student": {
            "name": "小宇",
            "grade": "九年级",
            "term_label": "九年级上学期 · 第 10 周",
            "streak_days": 23,
            "knowledge_lit_pct": overall_lit_pct,
            "mistake_freshness_pct": max(58, 71 - len(wrong_qs) * 2 + (2 if done_qs and not wrong_qs else 0)),
        },
        "tasks": {
            "homework": {
                "done_pages": len(caps),
                "target_pages": 4,
                "questions_done": len(done_qs),
                "questions_total": len(scored_qs),
            },
            "notes": {"last_score": 78},
            "vocab": {"recommend_count": 12, "unit": "九全 Unit 5"},
            "mistakes": {"due_count": 4 + len(wrong_qs), "session_wrong": len(wrong_qs)},
        },
        "subjects": {
            "math": {
                "subject": "math",
                "no": "01",
                "title": math_g["title"],
                "en_title": math_g["en_title"],
                "publisher": math_g["publisher"],
                "current": math_g["current"],
                "stats": math_g["stats"],
            },
            "en": {
                "subject": "en",
                "no": "02",
                "title": en_g["title"],
                "en_title": en_g["en_title"],
                "publisher": en_g["publisher"],
                "current": en_g["current"],
                "stats": en_g["stats"],
            },
        },
    }
