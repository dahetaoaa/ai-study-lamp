// Pure geometry: every point stays in original capture pixel coordinates.
// No DOM, image/model dependencies or global page-angle approximation.
const finitePoint = p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite);

export function validQuad(q) {
  if (!Array.isArray(q) || q.length !== 4 || !q.every(finitePoint)) return false;
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4];
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (Math.abs(cross) < 1e-8) return false;
    if (sign && Math.sign(cross) !== sign) return false;
    sign = Math.sign(cross);
  }
  return true;
}

export function bounds(points) {
  if (!points.length || !points.every(finitePoint)) return null;
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const x = Math.min(...xs), y = Math.min(...ys);
  return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
}

export function questionQuad(q) {
  if (validQuad(q.quad)) return q.quad.map(p => [...p]);
  if (Array.isArray(q.bbox) && q.bbox.length === 4 && q.bbox.every(Number.isFinite)) {
    const [x, y, w, h] = q.bbox;
    if (w > 0 && h > 0) return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
  }
  return null;
}

export function quadSize(q) {
  const length = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  return { w: (length(q[0], q[1]) + length(q[3], q[2])) / 2,
    h: (length(q[0], q[3]) + length(q[1], q[2])) / 2 };
}

/** Map the unit square's TL/TR/BR/BL to a genuine projective quadrilateral. */
export function quadProjector(q) {
  if (!validQuad(q)) return null;
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = q;
  const dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
  const dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
  let g = 0, h = 0;
  if (Math.abs(dx3) + Math.abs(dy3) > 1e-10) {
    const determinant = dx1 * dy2 - dx2 * dy1;
    if (Math.abs(determinant) < 1e-10) return null;
    g = (dx3 * dy2 - dx2 * dy3) / determinant;
    h = (dx1 * dy3 - dx3 * dy1) / determinant;
  }
  const a = x1 - x0 + g * x1, b = x3 - x0 + h * x3;
  const d = y1 - y0 + g * y1, e = y3 - y0 + h * y3;
  return ([u, v]) => {
    const denominator = g * u + h * v + 1;
    if (Math.abs(denominator) < 1e-8) return null;
    const point = [(a * u + b * v + x0) / denominator, (d * u + e * v + y0) / denominator];
    return finitePoint(point) ? point : null;
  };
}

export function primaryAnswerQuad(q) {
  const areas = q.answer_quads || [];
  const index = Number.isInteger(q.primary_area_index) ? q.primary_area_index : 0;
  return validQuad(areas[index]) ? areas[index] : areas.find(validQuad) || null;
}

/** Backend mark_quad is the drawing box, not an extra anchor offset. */
export function markQuad(q, fallbackSize = 40) {
  if (validQuad(q.mark_quad)) return q.mark_quad;
  const area = primaryAnswerQuad(q);
  if (area) {
    const project = quadProjector(area), size = quadSize(area);
    const side = Math.max(12, Math.min(size.h * 0.85, fallbackSize));
    const gap = Math.max(3, size.h * 0.12);
    const x0 = 1 + gap / size.w, x1 = x0 + side / size.w;
    const y0 = 0.5 - side / size.h / 2, y1 = 0.5 + side / size.h / 2;
    const result = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(project);
    if (validQuad(result)) return result;
  }
  // Legacy snapshots without answer geometry: preserve the anchor but do not rotate it by page angle.
  if (finitePoint(q.mark_pos)) {
    const [x, y] = q.mark_pos, s = fallbackSize;
    return [[x, y - s / 2], [x + s, y - s / 2], [x + s, y + s / 2], [x, y + s / 2]];
  }
  return null;
}

export function pathOf(points, close = true) {
  if (!points?.length || !points.every(finitePoint)) return '';
  const number = value => String(Math.round(value * 1000) / 1000);
  return 'M' + points.map(p => p.map(number).join(' ')).join('L') + (close ? 'Z' : '');
}

export function projectPath(q, unitPoints, close = false) {
  const project = quadProjector(q);
  return project ? pathOf(unitPoints.map(project), close) : '';
}

export function ellipsePoints(steps = 64, inset = 0.06) {
  return Array.from({ length: steps }, (_, i) => {
    const a = i * Math.PI * 2 / steps;
    return [0.5 + (0.5 - inset) * Math.cos(a), 0.5 + (0.5 - inset) * Math.sin(a)];
  });
}

export function checkPoints() {
  const segments = [
    [[0.08, 0.48], [0.17, 0.56], [0.26, 0.72], [0.34, 0.87]],
    [[0.34, 0.87], [0.48, 0.53], [0.67, 0.25], [0.94, 0.06]],
  ];
  return segments.flatMap((p, segment) => Array.from({ length: 21 }, (_, i) => {
    const t = i / 20, u = 1 - t;
    return [0, 1].map(axis => u ** 3 * p[0][axis] + 3 * u ** 2 * t * p[1][axis]
      + 3 * u * t ** 2 * p[2][axis] + t ** 3 * p[3][axis]);
  }).slice(segment ? 1 : 0));
}

/** The mask hole is exactly the OCR question quadrilateral. Framing may include its mark separately. */
export function focusGeometry(q, fallbackSize = 40, includeMark = true) {
  const points = questionQuad(q);
  if (!points) return null;
  const mark = includeMark ? markQuad(q, fallbackSize) : null;
  return { pts: points, box: bounds(mark ? [...points, ...mark] : points) };
}
