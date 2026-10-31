const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const {
  findMentionContext,
  findCommandContext,
  insertCommand,
  parseSkillInvocation,
  extractMentionPaths,
} = require('../lib/workspace_mentions')

const ROOT = path.join(__dirname, '..')
const renderer = fs.readFileSync(path.join(ROOT, 'renderer.js'), 'utf8')

test('a leading /command is completed from the word under the caret only', () => {
  const text = '/sk @report.pdf สรุปให้หน่อย'
  const ctx = findCommandContext(text, 3)
  assert.deepEqual(ctx, { prefix: '/sk', end: 3 })
  // caret inside a later word is still the same leading command token
  assert.equal(findCommandContext('/research find news', 5).prefix, '/research')
  assert.equal(findCommandContext('plain text', 3), null)
  assert.equal(findCommandContext('/abc', undefined).prefix, '/abc')
})

test('choosing a command replaces only the leading token and keeps the rest', () => {
  const text = '/sk @report.pdf สรุปให้หน่อย'
  const out = insertCommand(text, 3, 'research')
  assert.equal(out.value, '/research @report.pdf สรุปให้หน่อย')
  assert.equal(out.caret, '/research '.length)
  assert.equal(insertCommand('/r', 2, 'research').value, '/research ')
})

test('an @ under the caret wins over command completion after a /skill', () => {
  const text = '/research ดูไฟล์ @rep'
  assert.ok(findMentionContext(text, text.length))
  assert.equal(findMentionContext('/res', 4), null)
})

test('parseSkillInvocation splits /skill from its instruction for a known skill only', () => {
  const skills = ['research', 'review']
  assert.deepEqual(parseSkillInvocation('/research ข่าววันนี้ @a.txt', skills),
    { name: 'research', instruction: 'ข่าววันนี้ @a.txt' })
  assert.deepEqual(parseSkillInvocation('  /REVIEW   check this  ', skills),
    { name: 'review', instruction: 'check this' })
  assert.equal(parseSkillInvocation('/research', skills), null)            // toggle path
  assert.equal(parseSkillInvocation('/research   ', skills), null)
  assert.equal(parseSkillInvocation('/unknown do it', skills), null)       // server reports it
  assert.equal(parseSkillInvocation('not a command', skills), null)
  assert.equal(parseSkillInvocation('/research x', []), null)
})

test('mentions inside a /skill instruction resolve against the Active Workspace files', () => {
  const files = [{ name: 'report.pdf', relative_path: 'docs/report.pdf' }]
  const paths = extractMentionPaths('/research สรุป @report.pdf', files)
  assert.deepEqual(paths, ['docs/report.pdf'])
})

test('renderer sends a one-shot skill as a query carrying its mentions, not as a mode toggle', () => {
  assert.match(renderer, /parseSkillInvocation\(q, /)
  assert.match(renderer, /payload\.skill_once = skillCall\.name/)
  assert.match(renderer, /\.findCommandContext\(val, caret\)/)
  assert.match(renderer, /WorkspaceMentions\.insertCommand\(/)
  // a known /skill with an instruction must not fall into the plain-command branch
  assert.match(renderer, /if \(q\.startsWith\('\/'\) && !skillCall\)/)
  // the instruction (not the raw "/skill ..." text) is what mentions are extracted from
  assert.match(renderer, /extractMentionPaths\(queryText, mentionFiles\)/)
})
