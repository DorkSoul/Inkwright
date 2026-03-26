'use strict';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
var index = null;          // full JSON index
var paragraphs = [];       // flat array from index.paragraphs
var chapters = [];         // array from index.chapters
var currentChapterIdx = -1;
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

/**
 * Binary search paragraphs array for the entry containing `time`.
 * Returns index into `paragraphs`, or -1.
 */
function findActiveParagraph(time) {
  var lo = 0, hi = paragraphs.length - 1, result = -1;
  while (lo <= hi) {
    var mid = (lo + hi) >> 1;
    var p = paragraphs[mid];
    if (time >= p.start_time && time < p.end_time) {
      return mid;
    } else if (time < p.start_time) {
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
 * Returns the chapter index value (not array index).
 */
function findActiveChapter(time) {
  var result = 0;
  for (var i = 0; i < chapters.length; i++) {
    if (chapters[i].start_time <= time) {
      result = i;
    } else {
      break;
    }
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
  var ch = chapters[chapterArrayIdx];
  var nextChStart = chapterArrayIdx + 1 < chapters.length
    ? chapters[chapterArrayIdx + 1].start_time
    : Infinity;

  paragraphs.forEach(function (p, i) {
    if (p.chapter_index !== ch.index) return;

    var div = document.createElement('div');
    div.className = 'paragraph';
    div.textContent = p.text;
    div.dataset.paraIdx = i;
    div.addEventListener('click', function () {
      audio.currentTime = p.start_time;
      audio.play();
    });
    paragraphArea.appendChild(div);
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

function updateParagraphHighlight(paraArrayIdx) {
  var divs = paragraphArea.querySelectorAll('.paragraph');
  divs.forEach(function (div) {
    var idx = parseInt(div.dataset.paraIdx, 10);
    var isActive = idx === paraArrayIdx;
    div.classList.toggle('active', isActive);
    if (isActive) {
      // Scroll into view if not already visible
      var rect = div.getBoundingClientRect();
      var areaRect = paragraphArea.getBoundingClientRect();
      if (rect.top < areaRect.top || rect.bottom > areaRect.bottom) {
        div.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }
  });
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

  // Active paragraph
  var paraIdx = findActiveParagraph(time);
  updateParagraphHighlight(paraIdx);
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
      paragraphs = data.paragraphs || [];
      chapters = data.chapters || [];

      // Set total time display from index (audio may not be loaded yet)
      timeTotal.textContent = formatTime(data.duration_seconds);
      seekBar.max = 1000;

      renderChapterList();

      // Show first chapter immediately
      currentChapterIdx = 0;
      if (chapters.length > 0) {
        updateChapterHighlight(0);
        renderParagraphsForChapter(0);
      }

      // Start highlight loop
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
  // Handled by interval; this is a fallback for coarse updates
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

  // Audio events
  audio.addEventListener('timeupdate', onAudioTimeUpdate);
  audio.addEventListener('durationchange', onAudioDurationChange);
  audio.addEventListener('ended', onAudioEnded);
  audio.addEventListener('play', function () { btnPlay.innerHTML = '&#9646;&#9646;'; });
  audio.addEventListener('pause', function () { btnPlay.textContent = '\u25B6'; });

  // Controls
  btnPlay.addEventListener('click', onPlayPause);
  btnPrev.addEventListener('click', prevChapter);
  btnNext.addEventListener('click', nextChapter);

  speedSelect.addEventListener('change', function () {
    audio.playbackRate = parseFloat(speedSelect.value);
  });

  // Seek bar
  seekBar.addEventListener('mousedown', onSeekStart);
  seekBar.addEventListener('touchstart', onSeekStart, { passive: true });
  seekBar.addEventListener('input', onSeekInput);
  seekBar.addEventListener('mouseup', onSeekEnd);
  seekBar.addEventListener('touchend', onSeekEnd);

  // Sidebar
  if (chaptersToggle) chaptersToggle.addEventListener('click', openSidebar);
  if (sidebarClose)   sidebarClose.addEventListener('click', closeSidebar);
  if (sidebarOverlay) sidebarOverlay.addEventListener('click', closeSidebar);

  // Load index
  if (window.INKWRIGHT_INDEX_URL) {
    loadIndex(window.INKWRIGHT_INDEX_URL);
  }
});
