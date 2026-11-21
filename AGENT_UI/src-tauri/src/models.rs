//! The Model page's bridge to tools/models.py (run with the agent's `mlx` python: it imports Agent TH's model_slots.py).

use crate::agent;
use serde_json::{json, Value};
use std::process::{Command, Stdio};

pub fn run(args: &[&str]) -> Value {
    let script = agent::ui_dir().join("tools/models.py");
    let out = Command::new(agent::python())
        .arg(&script)
        .args(args)
        .env("AGENT_TH_DIR", agent::agent_dir())
        .env("PATH", agent::child_path())
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output();
    match out {
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout);
            text.lines()
                .rev()
                .find_map(|l| serde_json::from_str::<Value>(l.trim()).ok())
                .unwrap_or_else(|| json!({"ok": false, "error": "models.py gave no answer"}))
        }
        Err(e) => json!({"ok": false, "error": format!("could not run models.py: {e}")}),
    }
}
