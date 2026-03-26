'use strict';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
var index = null;
var chunks = [];          // [{paragraph_index, chapter_index, char_start, char_end, start_time, end_time}]
var sourceParas = [];     // [{paragraph_index, chapter_index, chapter_title, text}]
var sourceParaMap = {};   // paragraph_index → sourceParas entry
var chapters = [];
var currentChapterIdx = -1;
var currentChunkIdx = -1;   // index into chunks[]
var currentParaIdx = -1;    // paragraph_index of the active source paragraph
var isSeeking = false;
var highlightTimer = null;

// ---------------------------------------------------------------------------
// DOM refs (set after DOMContentLoaded)
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
  var ss = s % 60;
  return m + ':' + (ss < 10 ? '0' : '') + ss;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/**
 * Binary search chunks array for the entry containing `time`.
 * Returns index into chunks[], or -1.
 */
function findActiveChunk(time) {
  var lo = 0, hi = chunks.length - 1, result = -1;
  while (lo <= hi) {
    var mid = (lo + hi) >> 1;
    var c = chunks[mid];
    if (time >= c.start_time && time < c.end_time) {
      return mid;
    } else if (time < c.start_time) {
      hi = mid - 1;
    } else {
      result = mid;
      lo = mid + 1;
    }
  }
  return result;
}

/**
 * Binary search chapters array for the chapter active at `time`.
 */
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
  chapters.forEach(function (ch, i) {
    var li = document.createElement('li');
    li.className = 'chapter-item';
    li.textContent = ch.title;
    li.dataset.chapterArrayIdx = i;
    li.addEventListener('click', function () {
      audio.currentTime = ch.start_time;
      audio.play();
      closeSidebar();
    });
    chapterList.appendChild(li);
  });
}

function renderParagraphsForChapter(chapterArrayIdx) {
  paragraphArea.innerHTML = '';
  currentParaIdx = -1;   // force highlight refresh on next tick

  var ch = chapters[chapterArrayIdx];
  if (!ch) return;

  var parasForChapter = sourceParas.filter(function (p) {
    return p.chapter_index === ch.index;
  });

  parasForChapter.forEach(function (p) {
    var el = document.createElement('p');
    el.className = 'source-para';
    el.dataset.paraIdx = p.paragraph_index;
    el.textContent = p.text;

    // Clicking a paragraph seeks to its first chunk
    el.addEventListener('click', function () {
      var firstChunk = chunks.find(function (c) {
        return c.paragraph_index === p.paragraph_index;
      });
      if (firstChunk) {
        audio.currentTime = firstChunk.start_time;
        audio.play();
      }
    });

    paragraphArea.appendChild(el);
  });
}

function updateChapterHighlight(arrayIdx) {
  var items = chapterList.querySelectorAll('.chapter-item');
  items.forEach(function (li, i) {
    li.classList.toggle('active', i === arrayIdx);
  });
  if (items[arrayIdx]) {
    items[arrayIdx].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  if (chapters[arrayIdx]) {
    chapterTitle.textContent = chapters[arrayIdx].title;
  }
}

/**
 * Apply an inline <mark> highlight to the active chunk within its paragraph.
 * Clears the highlight from the previously active paragraph if it changed.
 */
function updateChunkHighlight(chunkIdx) {
  if (chunkIdx < 0 || chunkIdx >= chunks.length) {
    // Nothing active — clear any existing highlight
    if (currentParaIdx !== -1) {
      clearParaHighlight(currentParaIdx);
      currentParaIdx = -1;
    }
    currentChunkIdx = -1;
    return;
  }

  var chunk = chunks[chunkIdx];
  var paraIdx = chunk.paragraph_index;

  // If paragraph changed, clear old highlight
  if (paraIdx !== currentParaIdx && currentParaIdx !== -1) {
    clearParaHighlight(currentParaIdx);
  }

  // Apply highlight to the new/current paragraph
  var el = paragraphArea.querySelector('[data-para-idx="' + paraIdx + '"]');
  if (el) {
    var para = sourceParaMap[paraIdx];
    if (para) {
      var text = para.text;
      var s = chunk.char_start;
      var e = chunk.char_end;
      el.innerHTML =
        escapeHtml(text.slice(0, s)) +
        '<mark>' + escapeHtml(text.slice(s, e)) + '</mark>' +
        escapeHtml(text.slice(e));
    }
    el.classList.add('active');

    // Scroll into view if the paragraph just became active
    if (paraIdx !== currentParaIdx) {
      var rect = el.getBoundingClientRect();
      var areaRect = paragraphArea.getBoundingClientRect();
      if (rect.top < areaRect.top || rect.bottom > areaRect.bottom) {
        el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }
  }

  currentChunkIdx = chunkIdx;
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
// Poll loop (500 ms)
// ---------------------------------------------------------------------------
function onHighlightTick() {
  if (!index || isSeeking) return;

  var time = audio.currentTime;
  var duration = audio.duration || index.duration_seconds || 1;

  // Update seek bar
  seekBar.value = Math.round((time / duration) * 1000);

  // Update time displays
  timeCurrent.textContent = formatTime(time);

  // Update progress fill (thin bar at bottom of controls)
  progressFill.style.width = ((time / duration) * 100).toFixed(2) + '%';

  // Active chapter
  var chArrayIdx = findActiveChapter(time);
  if (chArrayIdx !== currentChapterIdx) {
    currentChapterIdx = chArrayIdx;
    updateChapterHighlight(chArrayIdx);
    renderParagraphsForChapter(chArrayIdx);
  }

  // Active chunk → highlight within paragraph
  var chunkIdx = findActiveChunk(time);
  if (chunkIdx !== currentChunkIdx) {
    updateChunkHighlight(chunkIdx);
  }
}

// ---------------------------------------------------------------------------
// Load index
// ---------------------------------------------------------------------------
function loadIndex(url) {
  fetch(url)
    .then(function (r) {
      if (!r.ok) throw new Error('Failed to load index: ' + r.status);
      return r.json();
    })
    .then(function (data) {
      index = data;
      chapters = data.chapters || [];

      // Support both new format (source_paragraphs + chunks)
      // and old format (paragraphs with text) for backward compatibility.
      if (data.source_paragraphs && data.chunks) {
        sourceParas = data.source_paragraphs;
        chunks = data.chunks;
      } else {
        // Old format: each paragraph entry is also its own chunk
        sourceParas = (data.paragraphs || []).map(function (p, i) {
          return {
            paragraph_index: i,
            chapter_index: p.chapter_index,
            chapter_title: '',
            text: p.text,
          };
        });
        chunks = (data.paragraphs || []).map(function (p, i) {
          return {
            paragraph_index: i,
            chapter_index: p.chapter_index,
            char_start: 0,
            char_end: (p.text || '').length,
            start_time: p.start_time,
            end_time: p.end_time,
          };
        });
      }

      // Build lookup map
      sourceParaMap = {};
      sourceParas.forEach(function (p) {
        sourceParaMap[p.paragraph_index] = p;
      });

      timeTotal.textContent = formatTime(data.duration_seconds);
      seekBar.max = 1000;

      renderChapterList();

      currentChapterIdx = 0;
      if (chapters.length > 0) {
        updateChapterHighlight(0);
        renderParagraphsForChapter(0);
      }

      highlightTimer = setInterval(onHighlightTick, 500);
    })
    .catch(function (e) {
      console.error(e);
      paragraphArea.textContent = 'Error loading index. Please refresh.';
    });
}

// ---------------------------------------------------------------------------
// Audio event handlers
// ---------------------------------------------------------------------------
function onAudioTimeUpdate() {
  // Handled by interval
}

function onAudioDurationChange() {
  if (audio.duration && isFinite(audio.duration)) {
    timeTotal.textContent = formatTime(audio.duration);
  }
}

function onAudioEnded() {
  btnPlay.textContent = '\u25B6';
}

function onPlayPause() {
  if (audio.paused) {
    audio.play();
    btnPlay.innerHTML = '&#9646;&#9646;';
  } else {
    audio.pause();
    btnPlay.textContent = '\u25B6';
  }
}

// ---------------------------------------------------------------------------
// Seek bar
// ---------------------------------------------------------------------------
function onSeekStart() {
  isSeeking = true;
}

function onSeekInput() {
  var duration = audio.duration || (index && index.duration_seconds) || 0;
  var t = (seekBar.value / 1000) * duration;
  timeCurrent.textContent = formatTime(t);
}

function onSeekEnd() {
  var duration = audio.duration || (index && index.duration_seconds) || 0;
  var t = (seekBar.value / 1000) * duration;
  audio.currentTime = t;
  isSeeking = false;
}

// ---------------------------------------------------------------------------
// Chapter prev / next
// ---------------------------------------------------------------------------
function prevChapter() {
  var target = currentChapterIdx - 1;
  if (target < 0) target = 0;
  audio.currentTime = chapters[target].start_time;
  if (audio.paused) audio.play();
}

function nextChapter() {
  var target = currentChapterIdx + 1;
  if (target >= chapters.length) return;
  audio.currentTime = chapters[target].start_time;
  if (audio.paused) audio.play();
}

// ---------------------------------------------------------------------------
// Sidebar (mobile drawer)
// ---------------------------------------------------------------------------
function openSidebar() {
  sidebar.classList.add('open');
  sidebarOverlay.classList.add('open');
}

function closeSidebar() {
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('open');
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function () {
  audio           = document.getElementById('audio');
  seekBar         = document.getElementById('seek-bar');
  btnPlay         = document.getElementById('btn-play');
  btnPrev         = document.getElementById('btn-prev-chapter');
  btnNext         = document.getElementById('btn-next-chapter');
  speedSelect     = document.getElementById('speed-select');
  timeCurrent     = document.getElementById('time-current');
  timeTotal       = document.getElementById('time-total');
  progressFill    = document.getElementById('progress-track-fill');
  chapterList     = document.getElementById('chapter-list');
  paragraphArea   = document.getElementById('paragraph-area');
  chapterTitle    = document.getElementById('player-chapter-title');
  chaptersToggle  = document.getElementById('chapters-toggle');
  sidebar         = document.getElementById('player-sidebar');
  sidebarOverlay  = document.getElementById('sidebar-overlay');
  sidebarClose    = document.getElementById('sidebar-close');

  audio.addEventListener('timeupdate', onAudioTimeUpdate);
  audio.addEventListener('durationchange', onAudioDurationChange);
  audio.addEventListener('ended', onAudioEnded);
  audio.addEventListener('play', function () { btnPlay.innerHTML = '&#9646;&#9646;'; });
  audio.addEventListener('pause', function () { btnPlay.textContent = '\u25B6'; });

  btnPlay.addEventListener('click', onPlayPause);
  btnPrev.addEventListener('click', prevChapter);
  btnNext.addEventListener('click', nextChapter);

  speedSelect.addEventListener('change', function () {
    audio.playbackRate = parseFloat(speedSelect.value);
  });

  seekBar.addEventListener('mousedown', onSeekStart);
  seekBar.addEventListener('touchstart', onSeekStart, { passive: true });
  seekBar.addEventListener('input', onSeekInput);
  seekBar.addEventListener('mouseup', onSeekEnd);
  seekBar.addEventListener('touchend', onSeekEnd);

  if (chaptersToggle) chaptersToggle.addEventListener('click', openSidebar);
  if (sidebarClose)   sidebarClose.addEventListener('click', closeSidebar);
  if (sidebarOverlay) sidebarOverlay.addEventListener('click', closeSidebar);

  if (window.INKWRIGHT_INDEX_URL) {
    loadIndex(window.INKWRIGHT_INDEX_URL);
  }
});
