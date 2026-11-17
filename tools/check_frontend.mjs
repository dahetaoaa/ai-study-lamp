// Run with: gjs -m tools/check_frontend.mjs
// Module loading and state regression checks only; not a browser/visual test.
// GJS already exposes window as a read-only alias of the global object.
globalThis.matchMedia = () => ({ matches: false });

const { S, emptyData, applyServerEvent, captureStats, viewStatus } = await import('../frontend/js/store.js');
await import('../frontend/js/panel.js');
await import('../frontend/js/sidebar.js');
await import('../frontend/js/stage.js');
await import('../frontend/js/capture-controls.js');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
S.d = emptyData();
S.d.selected_question_uid = 'old';
applyServerEvent(S.d, { type: 'capture.taken', capture_id: 'current' });
assert(S.d.selected_question_uid === null, 'Capture must clear old selection');
applyServerEvent(S.d, {
  type: 'page.matched', capture_id: 'current', questions: [
    { question_uid: 'q1', capture_id: 'current', qtype: 'choice', status: 'judging' },
    { question_uid: 'q2', capture_id: 'current', qtype: 'choice', status: 'right' },
  ],
});
applyServerEvent(S.d, {
  type: 'question.verdict', capture_id: 'current', question_uid: 'q1',
  status: 'uncertain', source: 'cloud_llm', model_answer: null,
});
assert(viewStatus(S.d.questions.q1) === 'uncertain', 'Uncertain state must survive merge');
const stats = captureStats(S.d.captures.current);
assert(stats.uncertain === 1 && stats.wrong === 0 && stats.right === 1, 'Uncertain must not count as wrong');
assert(stats.done === 2, 'Uncertain is a completed response, not still judging');
applyServerEvent(S.d, {
  type: 'question.explain_done', capture_id: 'current', question_uid: 'q1',
  full_text: '请补充题目条件', board_steps: [],
});
assert(S.d.questions.q1.status === 'uncertain', 'Explanation must not overwrite status');
const countBeforeReuse = Object.keys(S.d.questions).length;
S.d.selected_question_uid = 'q1';
applyServerEvent(S.d, { type: 'capture.reused', capture_id: 'current' });
assert(Object.keys(S.d.questions).length === countBeforeReuse, 'Reuse must not add questions');
assert(S.d.selected_question_uid === null && S.d.active_capture_id === 'current', 'Reuse must select existing capture');
applyServerEvent(S.d, { type: 'recording.state', active: true, id: 'rec_012345678901', enabled: true });
assert(S.d.recording.active, 'Recording state must survive event merge');
print('Frontend modules loaded; 8 state regression checks passed.');
