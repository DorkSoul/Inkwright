'use strict';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
var index = null;
var words = [];           // [{paragraph_index, char_start, char_end, start_time, end_time}]
var sourceParas = [];     // [{paragraph_index, chapter_index, chapter_title, text}]
var sourceParaMap = {};   // paragraph_index → entry
var chapters = [];
var currentChapterIdx = -1;
var currentWordIdx    = -1;
var currentParaIdx    = -1;
var isSeeking = false;
var rafId = null;

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
var audio, seekBar, btnPlay, btnPrev, btnNext, speedSelect;
var timeCurrent, timeTotal, progressFill, chapterList, paragraphArea;
var chapterTitle, chaptersToggle, sidebar, sidebarOverlay, sidebarClose;

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
function formatTime(seconds) {
  if (!isFinite(seconds)) return '0:00';
  var s = Math.floor(seconds);
  var m = Math.floor(s / 60);
  return m + ':' + ((s % 60 < 10) ? '0' : '') + (s % 60);
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Binary search words/chunks array for the entry active at `time`.
function findActiveWord(time) {
  var lo = 0, hi = words.length - 1, result = -1;
  while (lo <= hi) {
    var mid = (lo + hi) >> 1;
    var w = words[mid];
    if (time >= w.start_time && time < w.end_time) return mid;
    if (time < w.start_time) hi = mid - 1;
    else { result = mid; lo = mid + 1; }
  }
  return result;
}

function findActiveChapter(time) {
  var result = 0;
  for (var i = 0; i < chapters.length; i++) {
    if (chapters[i].start_time <= time) result = i;
    else break;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderChapterList() {
  chapterList.innerHTML = '';
  chapters.forEach(function(ch, i) {
    var li = document.createElement('li');
    li.className = 'chapter-item';
    li.textContent = ch.title;
    li.addEventListener('click', function() { audio.currentTime = ch.start_time; audio.play(); closeSidebar(); });
    chapterList.appendChild(li);
  });
}

function renderParagraphsForChapter(chapterArrayIdx) {
  paragraphArea.innerHTML = '';
  currentParaIdx = -1;
  currentWordIdx = -1;

  var ch = chapters[chapterArrayIdx];
  if (!ch) return;

  sourceParas.filter(function(p) { return p.chapter_index === ch.index; })
    .forEach(function(p) {
      var el = document.createElement('p');
      el.className = 'source-para';
      el.dataset.paraIdx = p.paragraph_index;
      el.textContent = p.text;
      el.addEventListener('click', function() {
        var first = words.find(function(w) { return w.paragraph_index === p.paragraph_index; });
        if (first) { audio.currentTime = first.start_time; audio.play(); }
      });
      paragraphArea.appendChild(el);
    });
}

function updateChapterHighlight(arrayIdx) {
  chapterList.querySelectorAll('.chapter-item').forEach(function(li, i) {
    li.classList.toggle('active', i === arrayIdx);
  });
  var item = chapterList.querySelectorAll('.chapter-item')[arrayIdx];
  if (item) item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  if (chapters[arrayIdx]) chapterTitle.textContent = chapters[arrayIdx].title;
}

function updateWordHighlight(wordIdx) {
  if (wordIdx < 0 || wordIdx >= words.length) {
    if (currentParaIdx !== -1) clearParaHighlight(currentParaIdx);
    currentParaIdx = -1;
    currentWordIdx = -1;
    return;
  }

  var w = words[wordIdx];
  var paraIdx = w.paragraph_index;

  if (paraIdx !== currentParaIdx && currentParaIdx !== -1) {
    clearParaHighlight(currentParaIdx);
  }

  var el = paragraphArea.querySelector('[data-para-idx="' + paraIdx + '"]');
  if (el) {
    var para = sourceParaMap[paraIdx];
    if (para) {
      var t = para.text;
      el.innerHTML =
        escapeHtml(t.slice(0, w.char_start)) +
        '<mark>' + escapeHtml(t.slice(w.char_start, w.char_end)) + '</mark>' +
        escapeHtml(t.slice(w.char_end));
    }
    el.classList.add('active');

    if (paraIdx !== currentParaIdx) {
      var rect = el.getBoundingClientRect();
      var areaRect = paragraphArea.getBoundingClientRect();
      if (rect.top < areaRect.top || rect.bottom > areaRect.bottom) {
        el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }
  }

  currentWordIdx = wordIdx;
  currentParaIdx = paraIdx;
}

function clearParaHighlight(paraIdx) {
  var el = paragraphArea.querySelector('[data-para-idx="' + paraIdx + '"]');
  if (el) {
    var para = sourceParaMap[paraIdx];
    if (para) el.textContent = para.text;
    el.classList.remove('active');
  }
}

// ---------------------------------------------------------------------------
// RAF highlight loop (replaces setInterval — runs every animation frame)
// ---------------------------------------------------------------------------
function highlightLoop() {
  if (index && !isSeeking) {
    var time     = audio.currentTime;
    var duration = audio.duration || index.duration_seconds || 1;

    seekBar.value = Math.round((time / duration) * 1000);
    timeCurrent.textContent = formatTime(time);
    progressFill.style.width = ((time / duration) * 100).toFixed(2) + '%';

    var chIdx = findActiveChapter(time);
    if (chIdx !== currentChapterIdx) {
      currentChapterIdx = chIdx;
      updateChapterHighlight(chIdx);
      renderParagraphsForChapter(chIdx);
    }

    var wIdx = findActiveWord(time);
    if (wIdx !== currentWordIdx) {
      updateWordHighlight(wIdx);
    }
  }
  rafId = requestAnimationFrame(highlightLoop);
}

// ---------------------------------------------------------------------------
// Load index
// ---------------------------------------------------------------------------
function loadIndex(url) {
  fetch(url)
    .then(function(r) {
      if (!r.ok) throw new Error('Failed to load index: ' + r.status);
      return r.json();
    })
    .then(function(data) {
      index    = data;
      chapters = data.chapters || [];

      if (data.source_paragraphs && data.words) {
        // New format — word-level timestamps
        sourceParas = data.source_paragraphs;
        words       = data.words;
      } else if (data.source_paragraphs && data.chunks) {
        // Intermediate format — chunk-level
        sourceParas = data.source_paragraphs;
        words       = data.chunks;
      } else {
        // Legacy format — paragraphs array
        sourceParas = (data.paragraphs || []).map(function(p, i) {
          return { paragraph_index: i, chapter_index: p.chapter_index,
                   chapter_title: '', text: p.text };
        });
        words = (data.paragraphs || []).map(function(p, i) {
          return { paragraph_index: i, char_start: 0,
                   char_end: (p.text || '').length,
                   start_time: p.start_time, end_time: p.end_time };
        });
      }

      sourceParaMap = {};
      sourceParas.forEach(function(p) { sourceParaMap[p.paragraph_index] = p; });

      timeTotal.textContent = formatTime(data.duration_seconds);
      renderChapterList();
      currentChapterIdx = 0;
      if (chapters.length > 0) {
        updateChapterHighlight(0);
        renderParagraphsForChapter(0);
      }

      rafId = requestAnimationFrame(highlightLoop);
    })
    .catch(function(e) {
      console.error(e);
      paragraphArea.textContent = 'Error loading index. Please refresh.';
    });
}

// ---------------------------------------------------------------------------
// Audio events
// ---------------------------------------------------------------------------
function onAudioDurationChange() {
  if (audio.duration && isFinite(audio.duration)) timeTotal.textContent = formatTime(audio.duration);
}
function onAudioEnded() { btnPlay.textContent = '\u25B6'; }
function onPlayPause() {
  if (audio.paused) { audio.play(); btnPlay.innerHTML = '&#9646;&#9646;'; }
  else              { audio.pause(); btnPlay.textContent = '\u25B6'; }
}

// ---------------------------------------------------------------------------
// Seek bar
// ---------------------------------------------------------------------------
function onSeekStart() { isSeeking = true; }
function onSeekInput() {
  var d = audio.duration || (index && index.duration_seconds) || 0;
  timeCurrent.textContent = formatTime((seekBar.value / 1000) * d);
}
function onSeekEnd() {
  var d = audio.duration || (index && index.duration_seconds) || 0;
  audio.currentTime = (seekBar.value / 1000) * d;
  isSeeking = false;
}

// ---------------------------------------------------------------------------
// Chapter prev / next
// ---------------------------------------------------------------------------
function prevChapter() {
  var t = Math.max(0, currentChapterIdx - 1);
  audio.currentTime = chapters[t].start_time;
  if (audio.paused) audio.play();
}
function nextChapter() {
  var t = currentChapterIdx + 1;
  if (t < chapters.length) { audio.currentTime = chapters[t].start_time; if (audio.paused) audio.play(); }
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------
function openSidebar()  { sidebar.classList.add('open'); sidebarOverlay.classList.add('open'); }
function closeSidebar() { sidebar.classList.remove('open'); sidebarOverlay.classList.remove('open'); }

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function() {
  audio          = document.getElementById('audio');
  seekBar        = document.getElementById('seek-bar');
  btnPlay        = document.getElementById('btn-play');
  btnPrev        = document.getElementById('btn-prev-chapter');
  btnNext        = document.getElementById('btn-next-chapter');
  speedSelect    = document.getElementById('speed-select');
  timeCurrent    = document.getElementById('time-current');
  timeTotal      = document.getElementById('time-total');
  progressFill   = document.getElementById('progress-track-fill');
  chapterList    = document.getElementById('chapter-list');
  paragraphArea  = document.getElementById('paragraph-area');
  chapterTitle   = document.getElementById('player-chapter-title');
  chaptersToggle = document.getElementById('chapters-toggle');
  sidebar        = document.getElementById('player-sidebar');
  sidebarOverlay = document.getElementById('sidebar-overlay');
  sidebarClose   = document.getElementById('sidebar-close');

  audio.addEventListener('durationchange', onAudioDurationChange);
  audio.addEventListener('ended', onAudioEnded);
  audio.addEventListener('play',  function() { btnPlay.innerHTML = '&#9646;&#9646;'; });
  audio.addEventListener('pause', function() { btnPlay.textContent = '\u25B6'; });

  btnPlay.addEventListener('click', onPlayPause);
  btnPrev.addEventListener('click', prevChapter);
  btnNext.addEventListener('click', nextChapter);

  speedSelect.addEventListener('change', function() {
    audio.playbackRate = parseFloat(speedSelect.value);
  });

  seekBar.addEventListener('mousedown',  onSeekStart);
  seekBar.addEventListener('touchstart', onSeekStart, { passive: true });
  seekBar.addEventListener('input',      onSeekInput);
  seekBar.addEventListener('mouseup',    onSeekEnd);
  seekBar.addEventListener('touchend',   onSeekEnd);

  if (chaptersToggle) chaptersToggle.addEventListener('click', openSidebar);
  if (sidebarClose)   sidebarClose.addEventListener('click', closeSidebar);
  if (sidebarOverlay) sidebarOverlay.addEventListener('click', closeSidebar);

  if (window.INKWRIGHT_INDEX_URL) loadIndex(window.INKWRIGHT_INDEX_URL);
});
