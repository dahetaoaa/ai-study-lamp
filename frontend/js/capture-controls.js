// Temporary local recordings and always-visible auto-capture diagnostics.
import { S } from './store.js';
import { $, esc } from './util.js';

const REASONS = {
  starting: '检测器启动中', paused: '已暂停', waiting_paper: '等待试卷', moving: '画面移动中，请放稳',
  settling: '正在等待稳定', ready: '正在识别', candidate: '正在核对纸面',
  already_captured: '本页已识别，等待换页', duplicate: '内容相同，已复用原结果',
  scene_changed: '检测到换页', blurry: '画面模糊，请调整距离或光线',
  unmatched: '未匹配到已入库试卷', incomplete_page: '试卷未完全入镜',
  no_camera: '真实相机不可用', camera_stalled: '相机画面未更新', detector_error: '检测异常，正在重试',
};
const STOP = { user: '已停止', duration_limit: '达到时长上限', size_limit: '达到单段大小上限',
  disk_limit: '磁盘空间不足', shutdown: '服务重启前已保存', unclean_shutdown: '录制被异常中断',
  camera_stalled: '相机停止更新', camera_source_changed: '相机源中断或改变' };

export function createCaptureControls(app) {
  let busy = false, lastFinished = '';
  const auto = $('#btnAutoCapture'), record = $('#btnRecord'), mark = $('#btnRecordMark');
  const panel = $('#recordingsPanel');
  async function request(path, body) {
    const response = await fetch('api/debug/recordings' + path, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const value = await response.json();
    if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : '录制请求失败');
    return value;
  }
  function render() {
    const connected = S.conn === 'open';
    const enabled = !!S.d.mode?.auto_capture;
    auto.disabled = !connected;
    auto.textContent = '自动识别：' + (enabled ? '开' : '关');
    auto.setAttribute('aria-pressed', String(enabled));
    const fresh = Date.now() - S.cam.at < 3000;
    const reason = !enabled ? '已暂停' : !connected ? '连接断开' : !fresh ? '等待检测状态…' :
      (REASONS[S.cam.reason] || (S.cam.stability === 'locked' ? '本页已识别，等待换页' : '等待试卷'));
    $('#autoCaptureState').textContent = reason + (enabled && !S.d.mode.auto_grade ? ' · 自动批改已关' : '');
    const state = S.d.recording || {};
    const available = state.enabled || state.active;
    record.hidden = !available;
    panel.hidden = !available;
    record.disabled = !connected || busy;
    record.textContent = state.active ? '停止录制' : '录制样本';
    record.classList.toggle('recording-active', !!state.active);
    mark.hidden = !state.active;
    mark.disabled = !connected || busy;
    $('#recordState').textContent = state.active ?
      `● ${Math.floor((state.duration_ms || 0) / 1000)}秒 · ${state.frames || 0}帧 · ${state.id || ''}` :
      state.id ? `${STOP[state.reason] || state.reason || '已保存'} · ${state.id}` : '';
  }
  async function refresh() {
    if (S.mock || S.conn !== 'open') return;
    try {
      const result = await request('');
      S.d.recording = result.state;
      $('#recordingList').innerHTML = result.clips.length ? result.clips.map(clip => {
        if (!/^rec_[0-9a-f]{12}$/.test(clip.id)) return '';
        const base = 'api/debug/recordings/' + encodeURIComponent(clip.id) + '/';
        const links = clip.active ? '录制中…' : `<a href="${base}video.avi" download>下载视频</a> · <a href="${base}manifest.json" download>参数</a> · <a href="${base}events.jsonl" download>事件</a>`;
        return `<div class="recording-item"><b>${esc(clip.id)}</b> ${esc(clip.label || '')}<br>` +
          `<span>${esc(clip.started_at || '')} · ${Math.round((clip.duration_ms || 0) / 1000)}秒 · ${esc(STOP[clip.reason] || clip.reason || clip.status)}</span> · ${links}</div>`;
      }).join('') : '<p>还没有录制样本。</p>';
      render();
    } catch (error) { app.toast(error.message, 'alert'); }
  }
  async function action(path) {
    if (busy) return;
    busy = true;
    render();
    try {
      S.d.recording = await request(path, { label: $('#recordLabel').value });
      if (path === '/start') app.setView(null);
      if (path === '/mark') app.toast('已记录“放稳”时间点，不会强制抓拍');
      if (path === '/stop') { panel.open = true; await refresh(); }
    } catch (error) { app.toast(error.message, 'alert'); }
    finally { busy = false; render(); }
  }
  auto.addEventListener('click', () => {
    const next = !S.d.mode?.auto_capture;
    app.send({ type: 'session.set_mode', auto_capture: next, ...(next ? { auto_grade: true } : {}) });
  });
  record.addEventListener('click', () => action(S.d.recording?.active ? '/stop' : '/start'));
  mark.addEventListener('click', () => action('/mark'));
  $('#btnRefreshRecordings').addEventListener('click', refresh);
  panel.addEventListener('toggle', () => { if (panel.open) refresh(); });
  function event(state) {
    S.d.recording = state;
    if (state.active && S.ui.view) app.setView(null);
    if (state.id && !state.active && lastFinished !== state.id + state.status) {
      lastFinished = state.id + state.status;
      refresh();
    }
    render();
  }
  return { render, refresh, event };
}
