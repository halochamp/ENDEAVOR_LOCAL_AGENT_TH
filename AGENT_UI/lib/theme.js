// theme.js — applies the saved light/dark theme before first paint.
//
// Loaded as a blocking <script> in <head>, right after the hljs stylesheet
// <link>, so the parser sets html[data-theme] and swaps the hljs theme href
// before <body> renders (avoids a flash of the wrong theme on startup).
;(function () {
  var saved = null
  try {
    saved = localStorage.getItem('endeavor-theme')
  } catch (e) {}
  var theme = (saved === 'light') ? 'light' : 'dark'
  document.documentElement.setAttribute('data-theme', theme)
  var hljsLink = document.getElementById('hljs-theme')
  if (hljsLink) {
    hljsLink.href = theme === 'light'
      ? './node_modules/@highlightjs/cdn-assets/styles/github.min.css'
      : './node_modules/@highlightjs/cdn-assets/styles/github-dark.min.css'
  }
})()
