// gjs -m tools/check_overlay.mjs [absolute path to cases.json]
// Pure local geometry only: does not import Markdown, KaTeX, DOM or remote code.
import GLib from 'gi://GLib';
import { validQuad, bounds, questionQuad, quadProjector, quadSize, markQuad, focusGeometry, projectPath, ellipsePoints, checkPoints } from '../frontend/js/overlay-geometry.js';

let assertions = 0;
function assert(value, label) { if (!value) throw new Error(label); assertions++; }
function near(a, b, tolerance = 1e-7) { return Math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance; }
const unit = [[0, 0], [1, 0], [1, 1], [0, 1]];
const trapezoid = [[100, 100], [520, 65], [470, 360], [130, 330]];
assert(validQuad(trapezoid), 'valid perspective quadrilateral');
assert(!validQuad([[0, 0], [1, 1], [0, 1], [1, 0]]), 'reject bow tie');
assert(!validQuad([[0, 0], [1, 0], [2, 0], [0, 1]]), 'reject collinear quad');
const project = quadProjector(trapezoid);
unit.forEach((point, i) => assert(near(project(point), trapezoid[i]), `corner ${i} maps exactly`));
assert(!near(project([0.5, 0.5]), trapezoid.reduce((v, p) => [v[0] + p[0] / 4, v[1] + p[1] / 4], [0, 0]), 0.1), 'projective center is not arithmetic center');
const q = {quad: trapezoid, mark_pos: [9000, 9000], answer_quads: [trapezoid]};
assert(JSON.stringify(focusGeometry(q).pts) === JSON.stringify(trapezoid), 'mask is exact q.quad, not a padded page-angle box');
const drawing = [[530, 100], [570, 96], [568, 135], [530, 138]];
assert(markQuad({...q, mark_quad: drawing}) === drawing, 'backend mark_quad takes priority with no added offset');
const fallback = markQuad(q, 40);
assert(bounds(fallback).x < 700, 'legacy bad mark anchor ignored when answer geometry exists');
assert(questionQuad({bbox: [5, 6, 10, 20]})[2][1] === 26, 'legacy bbox fallback');
assert(!projectPath(trapezoid, checkPoints()).includes('NaN'), 'projected check has no NaN');
assert(projectPath(trapezoid, ellipsePoints(), true).endsWith('Z'), 'answer outline is closed');
assert(markQuad({mark_pos: [10, 20]}, 30)[0][1] === 5, 'legacy anchor-only fallback');

const file = ARGV[0] || '/home/ubur/dahetao/data/video_testset/recordings_20261117/cases.json';
const readJSON = path => { const [, bytes] = GLib.file_get_contents(path); return JSON.parse(new TextDecoder().decode(bytes)); };
let captures = 0, questions = 0, maxLegacyError = 0, maxMaskError = 0, largestCase = null;
if (GLib.file_test(file, GLib.FileTest.EXISTS)) {
  const sources = [...new Set(readJSON(file).filter(c => c.observed).map(c => c.observed))];
  for (const source of sources) {
    const sample = readJSON('/home/ubur/dahetao/' + source), cap = sample.capture;
    const [tl, tr] = cap.page_quad;
    const angle = Math.atan2(tr[1] - tl[1], tr[0] - tl[0]);
    const size = Math.max(22, Math.min(150, Math.hypot(tr[0]-tl[0], tr[1]-tl[1]) * 0.055));
    const rotate = (p, a) => [p[0]*Math.cos(a)-p[1]*Math.sin(a), p[0]*Math.sin(a)+p[1]*Math.cos(a)];
    for (const row of sample.questions) {
      if (!validQuad(row.quad)) continue;
      questions++;
      const focus = focusGeometry(row, size);
      row.quad.forEach((point, i) => {
        const error = Math.hypot(focus.pts[i][0]-point[0], focus.pts[i][1]-point[1]);
        maxMaskError = Math.max(maxMaskError, error);
        assert(error === 0, source + ': mask vertex differs');
      });
      const mapper = quadProjector(row.quad);
      unit.forEach((p,i) => assert(near(mapper(p),row.quad[i],1e-5),source+': real quad homography'));
      const local = row.quad.map(p => rotate(p, -angle));
      if (row.mark_pos) {
        const center = [row.mark_pos[0]+Math.cos(angle)*size*.55,row.mark_pos[1]+Math.sin(angle)*size*.55];
        for(const [x,y] of [[-1,-1],[1,-1],[1,1],[-1,1]]) {
          const d = rotate([x*size*.62,y*size*.62],angle);
          local.push(rotate([center[0]+d[0],center[1]+d[1]],-angle));
        }
      }
      const b = bounds(local), m=size*.3;
      const old = [[b.x-m,b.y-m],[b.x+b.w+m,b.y-m],[b.x+b.w+m,b.y+b.h+m],[b.x-m,b.y+b.h+m]].map(p=>rotate(p,angle));
      const legacyError = Math.max(...old.map((p,i)=>Math.hypot(p[0]-row.quad[i][0],p[1]-row.quad[i][1])));
      if(legacyError>maxLegacyError){maxLegacyError=legacyError;largestCase={source,uid:row.question_uid};}
      const mark=markQuad(row,size);
      if(mark) assert(validQuad(mark),source+': fallback mark geometry');
    }
    captures++;
  }
}
print(JSON.stringify({assertions,captures,questions,maxMaskError,maxLegacyVertexDeviation:Math.round(maxLegacyError*100)/100,largestCase}));
