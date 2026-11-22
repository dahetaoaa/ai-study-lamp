import numpy as np
import pytest

from app.vision.annotations import annotation_geometry
from app.vision.matcher import project, quad


@pytest.mark.parametrize("area,anchor", [
    ([521, 269, 110, 59], [1195, 107]),
    ([418, 1724, 92, 66], [1455, 1648]),
    ([1120, 894, 110, 58], [1487, 864]),
])
def test_bad_legacy_anchor_cannot_move_mark_away_from_answer(area, anchor):
    q = {"bbox": [200, 50, 1300, 2000], "answer_areas": [{"bbox": area}], "mark_anchor": anchor}
    result = annotation_geometry(q, np.eye(3), 1654, 2339)
    center = np.mean(result["mark_quad"], axis=0)
    x, y, w, h = area
    assert abs(center[1] - (y + h / 2)) < 1
    assert 0 < center[0] - (x + w) < 80
    assert not np.allclose(result["mark_pos"], anchor)


def test_primary_area_and_ink_quad_share_projective_transform():
    areas = [[0, 270, 110, 70], [440, 278, 105, 66]]
    ink = [455, 290, 35, 35]
    q = {"bbox": [103, 282, 1165, 182], "answer_areas": [{"bbox": a} for a in areas]}
    h = np.array([[.62, .03, 40], [-.04, .59, 70], [.00006, -.00004, 1]])
    result = annotation_geometry(q, h, 1654, 2339, primary_area_index=1, ink_bbox=ink)
    assert result["primary_area_index"] == 1
    np.testing.assert_allclose(result["ink_quad"], project(quad(ink), h), atol=1e-4)
    local = np.array(project(result["mark_quad"], np.linalg.inv(h)))
    assert abs(local[0, 1] - local[1, 1]) < 1e-3
    assert abs(local[1, 0] - local[2, 0]) < 1e-3
    assert 0 < local[:, 0].mean() - (ink[0] + ink[2]) < 80


def test_margin_and_page_edge_markers_remain_on_paper():
    for area in ([0, 270, 110, 70], [1580, 2200, 70, 66]):
        q = {"bbox": [103, 282, 1165, 182], "answer_areas": [{"bbox": area}]}
        result = annotation_geometry(q, np.eye(3), 1654, 2339)
        points = np.array(result["mark_quad"])
        assert np.all(points >= 0)
        assert max(points[:, 0]) <= 1654 and max(points[:, 1]) <= 2339
    margin = annotation_geometry({"bbox": [103, 282, 1165, 182], "answer_areas": [{"bbox": [0, 270, 110, 70]}]}, np.eye(3), 1654, 2339)
    assert max(p[0] for p in margin["mark_quad"]) < 103  # 不盖住印刷题号


def test_outside_ink_bbox_is_not_used():
    q = {"bbox": [200, 200, 800, 100], "answer_areas": [{"bbox": [400, 200, 100, 60]}]}
    result = annotation_geometry(q, np.eye(3), 1654, 2339, ink_bbox=[1400, 1800, 80, 50])
    np.testing.assert_allclose(result["ink_quad"], quad(q["answer_areas"][0]["bbox"]))
