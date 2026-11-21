//! The agent side of the shell, ported 1:1 from the old Electron main.js: where things are, the per-launch token, and
//! starting/supervising `agent_server.py` on :8765. The model server is not the shell's business: Agent TH's own Python
//! (`model_runtime.py`, started by agent_server.py) owns its lifecycle, watchdog and the read-only MAX sharing rule; closing
//! this app stops only the agent server, never the model server.

use base64::Engine as _;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

pub const AGENT_PORT: u16 = 8765;
const MAX_CRASHES: u32 = 20;
const CRASH_WINDOW: Duration = Duration::from_secs(300);

/// Where the Agent TH checkout is: AGENT_TH_DIR, else the folder above this UI that holds agent_server.py (found from the
/// running program, so a moved checkout still works), else the source tree this was built from.
pub fn agent_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("AGENT_TH_DIR") {
        return PathBuf::from(d);
    }
    if let Ok(exe) = std::env::current_exe() {
        for dir in exe.ancestors() {
            if dir.join("agent_server.py").is_file() && dir.join("model_runtime.py").is_file() {
                return dir.to_path_buf();
            }
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// This UI's own folder (tools/models.py lives there).
pub fn ui_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        for dir in exe.ancestors() {
            if dir.join("tools/models.py").is_file() && dir.join("ui/index.html").is_file() {
                return dir.to_path_buf();
            }
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

pub fn workspace_dir() -> PathBuf {
    agent_dir().join("workspace")
}

/// The `mlx` conda env's python (MLX_PYTHON wins), as the old main.js resolved it: any conda install, never a fixed path.
pub fn python() -> PathBuf {
    if let Some(p) = std::env::var_os("MLX_PYTHON") {
        return PathBuf::from(p);
    }
    let env = std::env::var("MLX_CONDA_ENV").unwrap_or_else(|_| "mlx".into());
    let home = std::env::var("HOME").unwrap_or_default();
    for base in [
        format!("{home}/miniforge3"),
        format!("{home}/mambaforge"),
        format!("{home}/opt/anaconda3"),
        format!("{home}/anaconda3"),
        format!("{home}/miniconda3"),
        "/opt/homebrew/anaconda3".into(),
        "/opt/homebrew/miniforge3".into(),
        "/opt/homebrew/Caskroom/miniforge/base".into(),
        "/opt/anaconda3".into(),
        "/usr/local/anaconda3".into(),
    ] {
        let p = Path::new(&base).join("envs").join(&env).join("bin/python");
        if p.is_file() {
            return p;
        }
    }
    // any other install: ask conda itself where its envs are
    if let Ok(out) = Command::new("conda").args(["info", "--json"]).env("PATH", child_path_without_python()).stdin(Stdio::null()).stderr(Stdio::null()).output() {
        if let Ok(info) = serde_json::from_slice::<Value>(&out.stdout) {
            for e in info["envs"].as_array().into_iter().flatten().filter_map(|e| e.as_str()) {
                let p = Path::new(e).join("bin/python");
                if Path::new(e).file_name().map(|n| n.to_string_lossy() == env).unwrap_or(false) && p.is_file() {
                    return p;
                }
            }
        }
    }
    PathBuf::from("python3")
}

fn base_path() -> Vec<String> {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut parts: Vec<String> = vec!["/opt/homebrew/bin".into(), "/usr/local/bin".into(), format!("{home}/.local/bin"), format!("{home}/.cargo/bin"), format!("{home}/miniforge3/bin"), format!("{home}/anaconda3/bin"), "/opt/homebrew/anaconda3/bin".into()];
    for p in std::env::var("PATH").unwrap_or_default().split(':') {
        if !p.is_empty() && !parts.iter().any(|x| x == p) {
            parts.push(p.to_string());
        }
    }
    parts
}

fn child_path_without_python() -> String {
    base_path().join(":")
}

/// A Finder-opened app gets PATH=/usr/bin:/bin:...; the agent's tools call ffmpeg, git, uv... like they do from a terminal.
pub fn child_path() -> String {
    let mut parts = Vec::new();
    if let Some(bin) = python().parent() {
        parts.push(bin.display().to_string());
    }
    for p in base_path() {
        if !parts.contains(&p) {
            parts.push(p);
        }
    }
    parts.join(":")
}

pub fn new_token() -> String {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).expect("system random source");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn kill_port(port: u16) {
    let _ = Command::new("/bin/sh").arg("-c").arg(format!("lsof -ti:{port} | xargs kill -9 2>/dev/null")).status();
}

fn port_held(port: u16) -> bool {
    Command::new("/bin/sh")
        .arg("-c")
        .arg(format!("lsof -ti:{port} 2>/dev/null"))
        .output()
        .map(|o| !String::from_utf8_lossy(&o.stdout).trim().is_empty())
        .unwrap_or(false)
}

fn wait_port_free(port: u16, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if !port_held(port) {
            return true;
        }
        thread::sleep(Duration::from_millis(150));
    }
    false
}

/// Everything the shell keeps about the agent.
pub struct Agent {
    pub token: String,
    child: Mutex<Option<Child>>,
    pub pid: AtomicU32,
    pub quitting: AtomicBool,
    crashes: Mutex<(u32, Instant)>,
    pub startup_done: AtomicBool,
    pub last_status: Mutex<Value>,
    app: Mutex<Option<AppHandle>>,
}

impl Agent {
    pub fn new() -> Arc<Self> {
        Arc::new(Agent {
            token: new_token(),
            child: Mutex::new(None),
            pid: AtomicU32::new(0),
            quitting: AtomicBool::new(false),
            crashes: Mutex::new((0, Instant::now())),
            startup_done: AtomicBool::new(false),
            last_status: Mutex::new(Value::Null),
            app: Mutex::new(None),
        })
    }

    pub fn status(&self, msg: &str, phase: &str) {
        let payload = json!({"msg": msg, "phase": phase});
        *self.last_status.lock().unwrap() = payload.clone();
        if let Some(app) = self.app.lock().unwrap().as_ref() {
            let _ = app.emit("startup-status", payload);
        }
        eprintln!("[{phase}] {msg}");
    }

    /// The startup sequence of main.js: free :8765 (a stale agent holds an old token and cannot be adopted), start the agent.
    pub fn startup(self: &Arc<Self>, app: AppHandle) {
        *self.app.lock().unwrap() = Some(app.clone());
        let me = self.clone();
        thread::spawn(move || {
            me.status("Starting Agent TH runtime...", "info");
            kill_port(AGENT_PORT);
            wait_port_free(AGENT_PORT, Duration::from_secs(5));
            me.start_agent();
            thread::sleep(Duration::from_secs(2));
            me.status("Ready — connecting...", "ok");
            me.startup_done.store(true, Ordering::SeqCst);
            let _ = app.emit("startup-done", ());
        });
    }

    fn start_agent(self: &Arc<Self>) {
        if self.quitting.load(Ordering::SeqCst) {
            return;
        }
        let mut slot = self.child.lock().unwrap();
        if let Some(c) = slot.as_mut() {
            if matches!(c.try_wait(), Ok(None)) {
                return;
            }
        }
        let spawned = Command::new(python())
            .arg("agent_server.py")
            .current_dir(agent_dir())
            .env("AGENT_SERVER_TOKEN", &self.token)
            .env("AGENT_SERVER_PORT", AGENT_PORT.to_string())
            .env("PATH", child_path())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0) // its own group: the whole uvicorn tree is stopped together on quit
            .spawn();
        let mut child = match spawned {
            Ok(c) => c,
            Err(e) => {
                drop(slot);
                self.status(&format!("Agent server could not start: {e}"), "error");
                return;
            }
        };
        let pid = child.id();
        self.pid.store(pid, Ordering::SeqCst);
        for stream in [child.stdout.take().map(|s| Box::new(s) as Box<dyn Read + Send>), child.stderr.take().map(|s| Box::new(s) as Box<dyn Read + Send>)].into_iter().flatten() {
            thread::spawn(move || {
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    if !line.contains("INFO:") {
                        eprintln!("[agent] {line}");
                    }
                }
            });
        }
        *slot = Some(child);
        drop(slot);
        let me = self.clone();
        thread::spawn(move || me.watch_agent(pid));
    }

    /// The crash handler of main.js: back off, free the port, start again (model recovery is Python's own job).
    fn watch_agent(self: &Arc<Self>, pid: u32) {
        loop {
            thread::sleep(Duration::from_millis(500));
            let mut slot = self.child.lock().unwrap();
            let Some(child) = slot.as_mut() else { return };
            if child.id() != pid {
                return; // a newer child owns the slot
            }
            match child.try_wait() {
                Ok(None) => continue,
                Ok(Some(status)) => {
                    *slot = None;
                    drop(slot);
                    eprintln!("[agent] exited {status}");
                    break;
                }
                Err(_) => return,
            }
        }
        if self.quitting.load(Ordering::SeqCst) {
            return;
        }
        let count = {
            let mut c = self.crashes.lock().unwrap();
            if c.1.elapsed() > CRASH_WINDOW {
                *c = (0, Instant::now());
            }
            c.0 += 1;
            c.0
        };
        if count > MAX_CRASHES {
            self.status(&format!("Agent crashed {count} times — giving up. Restart the app to retry."), "error");
            return;
        }
        let delay = backoff_ms(count);
        self.status(&format!("Agent stopped (crash {count}/{MAX_CRASHES}) — retry in {}s…", delay / 1000), "warn");
        thread::sleep(Duration::from_millis(delay));
        kill_port(AGENT_PORT);
        wait_port_free(AGENT_PORT, Duration::from_secs(5));
        self.start_agent();
    }

    /// On quit: stop the agent's whole process group (uvicorn threads, tool subprocesses). The model server stays.
    pub fn shutdown(&self) {
        self.quitting.store(true, Ordering::SeqCst);
        let pid = self.pid.load(Ordering::SeqCst);
        if pid > 0 {
            unsafe {
                libc::kill(-(pid as i32), libc::SIGTERM);
            }
        }
    }
}

/// min(3000 * 2^(n-1), 30000) ms, like main.js.
fn backoff_ms(count: u32) -> u64 {
    (3000u64 * 2u64.pow(count.saturating_sub(1).min(10))).min(30_000)
}

#[cfg(test)]
mod tests {
    #[test]
    fn backoff_doubles_then_caps() {
        assert_eq!(super::backoff_ms(1), 3000);
        assert_eq!(super::backoff_ms(2), 6000);
        assert_eq!(super::backoff_ms(4), 24_000);
        assert_eq!(super::backoff_ms(5), 30_000);
        assert_eq!(super::backoff_ms(20), 30_000);
    }
}
