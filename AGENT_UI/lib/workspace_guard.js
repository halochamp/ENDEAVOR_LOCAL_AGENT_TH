const fs = require('fs')
const path = require('path')

// File management is restricted to a workspace dir. Resolve real paths (follows
// symlinks) and require the target to sit inside workspaceDir — blocks path
// traversal and symlink escapes from deleting/opening anything outside.
function isInsideWorkspace(target, workspaceDir) {
  try {
    const ws = fs.realpathSync(workspaceDir)
    const real = fs.realpathSync(target)
    return real === ws || real.startsWith(ws + path.sep)
  } catch {
    return false
  }
}

module.exports = { isInsideWorkspace }
