import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.knowledge.service import get_overview_payload, get_subject_graph, load_base_graphs
from app.main import create_app


def test_knowledge_graphs_completeness():
    load_base_graphs.cache_clear()
    math = get_subject_graph("math")
    en = get_subject_graph("en")

    # 人教版初中数学 6 册 29 章完整覆盖
    assert len(math["books"]) == 6
    assert math["stats"]["chapters_count"] == 29
    assert math["stats"]["total_nodes"] >= 85
    assert math["current"]["section_code"] == "21.2.3"
    assert math["current"]["section_title"] == "因式分解法"
    assert len(math["edges"]) >= math["stats"]["total_nodes"] - 6

    # 人教版初中英语 Go for it! 5 册完整覆盖 + 语法专题集群
    assert len(en["books"]) == 5
    assert en["stats"]["total_nodes"] == 58
    assert en["current"]["unit_id"] == "en_9_u5"
    assert "What are the shirts made of?" in en["current"]["unit_title"]
    assert len(en["grammar_clusters"]) >= 8
    assert 1400 <= en["stats"]["vocab_total"] <= 1800


def test_knowledge_api_and_live_session_fusion(tmp_path):
    load_base_graphs.cache_clear()
    session_mock = {
        "captures": {"cap_1": {"subject": "math"}},
        "questions": {
            "q1": {
                "question_uid": "q1",
                "capture_id": "cap_1",
                "no": "1",
                "qtype": "choice",
                "status": "wrong",
                "tag": "因式分解法解一元二次方程",
            }
        },
    }
    overview = get_overview_payload(session_mock)
    assert overview["tasks"]["homework"]["done_pages"] == 1
    assert overview["tasks"]["mistakes"]["session_wrong"] == 1

    math_live = get_subject_graph("math", session_mock)
    sec_2123 = next(
        s
        for b in math_live["books"]
        for ch in b["chapters"]
        for s in ch["sections"]
        if s["code"] == "21.2.3"
    )
    assert sec_2123["session_practice"]["total"] == 1
    assert sec_2123["session_practice"]["wrong"] == 1

    app = create_app(
        Settings(
            _env_file=None,
            camera_backend="mock",
            auto_capture=False,
            tts_provider="mock",
            templates_dir=tmp_path,
        )
    )
    with TestClient(app) as client:
        r_ov = client.get("/api/knowledge/overview")
        assert r_ov.status_code == 200
        assert "math" in r_ov.json()["subjects"]
        assert "en" in r_ov.json()["subjects"]

        r_math = client.get("/api/knowledge/graph/math")
        assert r_math.status_code == 200
        assert r_math.json()["subject"] == "math"

        r_en = client.get("/api/knowledge/graph/en")
        assert r_en.status_code == 200
        assert r_en.json()["subject"] == "en"

        assert client.get("/api/knowledge/graph/physics").status_code == 404
