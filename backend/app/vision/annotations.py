"""批改几何只由实际答题区导出，再用同一个H投射；不使用离题的旧mark_anchor。"""
import math

from app.vision.matcher import project, quad


def annotation_geometry(question, homography, page_width, page_height, primary_area_index=0, ink_bbox=None):
    areas = question.get("answer_areas") or []
    if not isinstance(primary_area_index, int) or not 0 <= primary_area_index < len(areas):
        primary_area_index = 0
    region = areas[primary_area_index]["bbox"] if areas else question["bbox"]
    ax, ay, aw, ah = map(float, region)
    target = [ax, ay, aw, ah]
    if ink_bbox is not None and len(ink_bbox) == 4:
        values = list(map(float, ink_bbox))
        if all(math.isfinite(v) for v in values) and values[2] > 0 and values[3] > 0:
            # 墨迹必须与所选答题区相交，拒绝把其他题的印刷体坐标当成目标。
            ix, iy, iw, ih = values
            if (iw <= aw + 32 and ih <= ah + 32
                    and ix < ax + aw + 16 and ix + iw > ax - 16
                    and iy < ay + ah + 16 and iy + ih > ay - 16):
                target = values
    x, y, w, h = target
    size = min(68.0, max(28.0, ah * 0.78))
    gap = max(6.0, size * 0.15)
    bx = x + w + gap
    by = y + (h - size) / 2
    margin_area = ax <= 16 and ax + aw <= question["bbox"][0] + 20
    if margin_area and target == [ax, ay, aw, ah]:
        # 题号左侧空白答题槽：标记放槽内，不跨到印刷题干上。
        size = min(size, max(20.0, aw - 12))
        bx, by = ax + (aw - size) / 2, ay + (ah - size) / 2
    elif margin_area and x - size - gap >= 4:
        bx = x - size - gap
    if bx + size > page_width - 4:
        bx = x - gap - size
    bx = max(4.0, min(bx, page_width - size - 4))
    by = max(4.0, min(by, page_height - size - 4))
    box = [bx, by, size, size]
    return {
        "primary_area_index": primary_area_index,
        "mark_quad": project(quad(box), homography),
        "mark_pos": project([[bx, by + size / 2]], homography)[0],
        "ink_quad": project(quad(target), homography),
    }
