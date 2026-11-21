// Copies the three browser libraries the renderer uses from node_modules into ui/vendor/ (Tauri serves only ui/).
// Run by `npm run build` / `npm run dev`; ui/vendor/ is not committed.
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.join(__dirname, '..')
const OUT = path.join(ROOT, 'ui', 'vendor')
const FILES = [
  ['marked/lib/marked.umd.js', 'marked.umd.js'],
  ['dompurify/dist/purify.min.js', 'purify.min.js'],
  ['@highlightjs/cdn-assets/highlight.min.js', 'highlight.min.js'],
  ['@highlightjs/cdn-assets/styles/github.min.css', 'github.min.css'],
  ['@highlightjs/cdn-assets/styles/github-dark.min.css', 'github-dark.min.css'],
]

fs.mkdirSync(OUT, { recursive: true })
for (const [from, to] of FILES) {
  fs.copyFileSync(path.join(ROOT, 'node_modules', from), path.join(OUT, to))
}
console.log(`vendor: ${FILES.length} files -> ui/vendor`)
