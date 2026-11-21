//! One command per `window.electronAPI` method (ui/bridge.js), with the same argument clamping and answers as the old
//! Electron main.js. Native dialogs run on a blocking thread, never on the IPC thread. The agent backend stays the policy
//! authority for edit access: this shell asks it (`/edit-access/authorize`, `/edit-access/delete`) before opening or deleting.

use crate::agent::{self, Agent};
use crate::http::{self, encode_component, Upload};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, LogicalSize, Manager, PhysicalPosition, PhysicalSize, State};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_opener::OpenerExt;

/// Reading speed when the page sends none: the agent's `text_to_speech.DEFAULT_SPEED` (owner 2026-11-21).
const DEFAULT_TTS_SPEED: f64 = 1.15;

type AgentState<'a> = State<'a, Arc<Agent>>;

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    tauri::async_runtime::spawn_blocking(f).await.expect("blocking task")
}

fn picked(p: tauri_plugin_dialog::FilePath) -> Option<PathBuf> {
    p.into_path().ok()
}

#[tauri::command]
pub fn startup_state(agent: AgentState<'_>) -> Value {
    json!({"done": agent.startup_done.load(std::sync::atomic::Ordering::SeqCst), "status": *agent.last_status.lock().unwrap()})
}

#[tauri::command]
pub fn get_token(agent: AgentState<'_>) -> String {
    agent.token.clone()
}

#[tauri::command]
pub fn open_workspace(app: AppHandle) {
    let _ = app.opener().open_path(agent::workspace_dir().display().to_string(), None::<&str>);
}

/// Web links leave the app for the browser; nothing else does.
#[tauri::command]
pub fn open_url(app: AppHandle, url: String) {
    if url.starts_with("http://") || url.starts_with("https://") {
        let _ = app.opener().open_url(url, None::<&str>);
    }
}

/// Runtime selection stays behind this boundary: the page receives only the agent's answer, never the token.
#[tauri::command]
pub async fn set_runtime_settings(agent: AgentState<'_>, payload: Value) -> Result<Value, ()> {
    if !payload.is_object() {
        return Ok(json!({"ok": false, "error": "invalid runtime settings"}));
    }
    let text = |k: &str| payload[k].as_str().map(String::from).unwrap_or_else(|| if payload[k].is_null() { String::new() } else { payload[k].to_string() });
    let number = |k: &str| payload[k].as_f64().or_else(|| payload[k].as_str().and_then(|v| v.parse().ok())).unwrap_or(f64::NAN);
    let mut body = json!({
        "model": text("model"),
        "thinking_budget": number("thinking_budget"),
        "server_mode": text("server_mode"),
        "server_port": number("server_port"),
    });
    if payload["confirmed_low_ram"].as_bool() == Some(true) || payload["confirmed_low_ram"].is_string() && !payload["confirmed_low_ram"].as_str().unwrap_or("").is_empty() {
        body["confirmed_low_ram"] = json!(true);
    }
    Ok(http::request(&agent.token, "POST", "/runtime-settings", Some(&body), None, Duration::from_secs(300)).await)
}

/// The job routes the panels poll / control (an allowlist: the page cannot reach any other agent route this way).
const JOB_ROUTES: [(&str, bool); 9] = [
    ("/pdf-to-text/status", false),
    ("/tts/status", false),
    ("/tts/pause", true),
    ("/tts/resume", true),
    ("/tts/cancel", true),
    ("/translate/status", false),
    ("/translate/pause", true),
    ("/translate/resume", true),
    ("/translate/cancel", true),
];

#[tauri::command]
pub async fn agent_job(agent: AgentState<'_>, route: String, job_id: String, post: bool) -> Result<Value, ()> {
    if !JOB_ROUTES.contains(&(route.as_str(), post)) {
        return Ok(json!({"ok": false, "error": "route not allowed"}));
    }
    let full = format!("{route}?job_id={}", encode_component(&job_id));
    Ok(http::request(&agent.token, if post { "POST" } else { "GET" }, &full, None, None, Duration::from_secs(300)).await)
}

#[tauri::command]
pub async fn agent_get(agent: AgentState<'_>, route: String) -> Result<Value, ()> {
    if route != "/tts/voices" {
        return Ok(json!({"ok": false, "error": "route not allowed"}));
    }
    Ok(http::request(&agent.token, "GET", &route, None, None, Duration::from_secs(300)).await)
}

async fn pick_one(app: &AppHandle, title: &'static str, filter: (&'static str, &'static [&'static str])) -> Option<PathBuf> {
    let app = app.clone();
    blocking(move || app.dialog().file().set_title(title).add_filter(filter.0, filter.1).blocking_pick_file().and_then(picked)).await
}

#[allow(clippy::too_many_arguments)]
async fn upload(agent: &Agent, route: &str, path: &Path, content_type: &str, mut headers: Vec<(&'static str, String)>, max: u64, empty_msg: &str, too_big: &str, timeout: Duration, timeout_msg: &str) -> Value {
    match std::fs::metadata(path) {
        Ok(m) if m.is_file() && m.len() > 0 => {
            if m.len() > max {
                return json!({"ok": false, "error": too_big});
            }
        }
        Ok(_) => return json!({"ok": false, "error": empty_msg}),
        Err(e) => return json!({"ok": false, "error": e.to_string()}),
    }
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    headers.push(("X-File-Name", encode_component(&name)));
    let mut out = http::request(&agent.token, "POST", route, None, Some(Upload { path, content_type, headers }), timeout).await;
    if out["error"] == json!("Agent request timed out") {
        out["error"] = json!(timeout_msg);
    }
    out
}

#[tauri::command]
pub async fn pdf_to_text_start(app: AppHandle, agent: AgentState<'_>, rewrite_thai: bool) -> Result<Value, ()> {
    let Some(path) = pick_one(&app, "เลือก PDF เพื่อแปลงเป็นข้อความ", ("PDF", &["pdf"])).await else {
        return Ok(json!({"ok": false, "cancelled": true}));
    };
    let headers = vec![("X-Rewrite-Thai", if rewrite_thai { "1" } else { "0" }.to_string())];
    Ok(upload(&agent, "/pdf-to-text/start", &path, "application/pdf", headers, u64::MAX, "PDF ว่างหรืออ่านไม่ได้", "", Duration::from_secs(300), "PDF upload timed out").await)
}

/// A voice id goes into an HTTP header: printable ASCII only, bounded. The agent still validates it against its voices.
fn clean_voice(v: &str) -> String {
    if (1..=64).contains(&v.len()) && v.bytes().all(|b| b.is_ascii_graphic()) {
        v.to_string()
    } else {
        String::new()
    }
}

#[tauri::command]
pub async fn tts_start(app: AppHandle, agent: AgentState<'_>, voice: String, speed: f64) -> Result<Value, ()> {
    let speed = if speed.is_finite() && speed != 0.0 { speed.clamp(0.8, 1.2) } else { DEFAULT_TTS_SPEED };
    let Some(path) = pick_one(&app, "เลือกไฟล์ข้อความเพื่อให้อ่านออกเสียง", ("Text / Markdown", &["txt", "md", "markdown"])).await else {
        return Ok(json!({"ok": false, "cancelled": true}));
    };
    let ct = if path.extension().map(|e| e.to_string_lossy().to_lowercase()) == Some("txt".into()) { "text/plain" } else { "text/markdown" };
    let headers = vec![("X-Voice", clean_voice(&voice)), ("X-Speed", fmt_number(speed))];
    Ok(upload(&agent, "/tts/start", &path, ct, headers, 8 * 1024 * 1024, "ไฟล์ว่างหรืออ่านไม่ได้", "ไฟล์ใหญ่เกิน 8MB", Duration::from_secs(120), "Text to Speech upload timed out").await)
}

/// JavaScript's String(number): 1 -> "1", 0.9 -> "0.9".
fn fmt_number(n: f64) -> String {
    if n.fract() == 0.0 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

const TRANSLATE_TYPES: [(&str, &str); 13] = [
    ("pdf", "application/pdf"),
    ("doc", "application/msword"),
    ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ("xls", "application/vnd.ms-excel"),
    ("html", "text/html"),
    ("htm", "text/html"),
    ("txt", "text/plain"),
    ("md", "text/markdown"),
    ("markdown", "text/markdown"),
    ("csv", "text/csv"),
    ("tsv", "text/tab-separated-values"),
    ("rtf", "application/rtf"),
];

#[tauri::command]
pub async fn translate_start(app: AppHandle, agent: AgentState<'_>) -> Result<Value, ()> {
    let exts: &'static [&'static str] = &["pdf", "doc", "docx", "txt", "md", "markdown", "log", "rst", "csv", "tsv", "html", "htm", "rtf", "xlsx", "xls"];
    let Some(path) = pick_one(&app, "เลือกเอกสารเพื่อแปลเป็นภาษาไทย", ("Documents", exts)).await else {
        return Ok(json!({"ok": false, "cancelled": true}));
    };
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    let ct = TRANSLATE_TYPES.iter().find(|(e, _)| *e == ext).map(|(_, t)| *t).unwrap_or("application/octet-stream");
    let headers = vec![("X-Target-Language", "th".to_string())];
    Ok(upload(&agent, "/translate/start", &path, ct, headers, 250 * 1024 * 1024, "ไฟล์ว่างหรืออ่านไม่ได้", "ไฟล์ใหญ่เกิน 250MB", Duration::from_secs(600), "Translate upload timed out").await)
}

/// Edit → the agent decides whether the path is inside the approved scope; only then is it opened in the OS default app.
/// Returns "" on success or an error string the page shows.
#[tauri::command]
pub async fn edit_file(app: AppHandle, agent: AgentState<'_>, path: String) -> Result<String, ()> {
    let body = json!({"path": path});
    let answer = http::request(&agent.token, "POST", "/edit-access/authorize", Some(&body), None, Duration::from_secs(300)).await;
    let Some(authorized) = answer["path"].as_str().filter(|p| !p.is_empty()) else {
        return Ok(answer["error"].as_str().unwrap_or("outside approved edit scope").to_string());
    };
    Ok(match app.opener().open_path(authorized.to_string(), None::<&str>) {
        Ok(()) => String::new(),
        Err(e) => e.to_string(),
    })
}

/// Delete → confirm in a native dialog (destructive), then the agent deletes inside the approved scope. Files only.
#[tauri::command]
pub async fn delete_file(app: AppHandle, agent: AgentState<'_>, path: String) -> Result<Value, ()> {
    let name = Path::new(&path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let app2 = app.clone();
    let yes = blocking(move || {
        app2.dialog()
            .message("การลบนี้ย้อนกลับไม่ได้")
            .title(format!("ลบไฟล์ \"{name}\"?"))
            .kind(MessageDialogKind::Warning)
            .buttons(MessageDialogButtons::OkCancelCustom("ลบ".into(), "ยกเลิก".into()))
            .blocking_show()
    })
    .await;
    if !yes {
        return Ok(json!({"deleted": false, "cancelled": true}));
    }
    let body = json!({"path": path});
    let answer = http::request(&agent.token, "POST", "/edit-access/delete", Some(&body), None, Duration::from_secs(300)).await;
    Ok(if answer["deleted"] == json!(true) {
        json!({"deleted": true})
    } else {
        json!({"deleted": false, "error": answer["error"].as_str().unwrap_or("outside approved edit scope")})
    })
}

/// One file or null (Attach).
#[tauri::command]
pub async fn show_open_dialog(app: AppHandle) -> Result<Option<String>, ()> {
    let one = blocking(move || app.dialog().file().blocking_pick_file()).await;
    Ok(one.and_then(picked).map(|p| p.display().to_string()))
}

/// Canonical existing paths of one kind; everything else is counted as rejected (main.js did the same).
fn canonical(paths: Vec<PathBuf>, want_dir: bool) -> Value {
    let mut out: Vec<String> = Vec::new();
    let mut rejected = 0;
    for p in paths {
        match std::fs::canonicalize(&p) {
            Ok(real) if if want_dir { real.is_dir() } else { real.is_file() } => {
                let s = real.display().to_string();
                if !out.contains(&s) {
                    out.push(s);
                }
            }
            _ => rejected += 1,
        }
    }
    json!({"paths": out, "rejected": rejected})
}

/// Pin selection may come from anywhere the user can browse; the backend revalidates every path on every turn.
#[tauri::command]
pub async fn show_pin_dialog(app: AppHandle) -> Result<Value, ()> {
    let picked_files = blocking(move || app.dialog().file().set_title("เลือกไฟล์เพื่อ Pin").blocking_pick_files()).await;
    Ok(canonical(picked_files.unwrap_or_default().into_iter().filter_map(picked).collect(), false))
}

/// Approved Edit / Active Workspace selection is directory-only; the backend validates every returned path again.
#[tauri::command]
pub async fn show_approved_edit_folder_dialog(app: AppHandle) -> Result<Value, ()> {
    let picked_dirs = blocking(move || app.dialog().file().set_title("เลือกโฟลเดอร์สำหรับสิทธิ์แก้ไข").blocking_pick_folders()).await;
    Ok(canonical(picked_dirs.unwrap_or_default().into_iter().filter_map(picked).collect(), true))
}

/// Compact: a small always-on-top chat window; leaving it restores the exact earlier size and position.
#[derive(Default)]
pub struct WindowMode {
    compact: Mutex<bool>,
    saved: Mutex<Option<(PhysicalPosition<i32>, PhysicalSize<u32>)>>,
}

#[tauri::command]
pub fn set_compact(app: AppHandle, state: State<'_, WindowMode>, compact: bool) {
    let Some(w) = app.get_webview_window("main") else { return };
    let mut current = state.compact.lock().unwrap();
    if compact {
        if !*current {
            if let (Ok(pos), Ok(size)) = (w.outer_position(), w.outer_size()) {
                *state.saved.lock().unwrap() = Some((pos, size));
            }
        }
        let _ = w.set_min_size(Some(LogicalSize::new(360.0, 420.0)));
        let _ = w.set_size(LogicalSize::new(420.0, 560.0));
        let _ = w.set_always_on_top(true);
    } else {
        let _ = w.set_always_on_top(false);
        let _ = w.set_min_size(Some(LogicalSize::new(crate::MIN_W, crate::MIN_H)));
        match state.saved.lock().unwrap().take() {
            Some((pos, size)) => {
                let _ = w.set_size(size);
                let _ = w.set_position(pos);
            }
            None => {
                let _ = w.set_size(LogicalSize::new(crate::DEFAULT_W, crate::DEFAULT_H));
            }
        }
    }
    *current = compact;
}

/// /exit from the page → confirm natively, then quit. Quitting stops this app's agent server only; the model server is
/// independent owner state and stays until Stop in Settings or agent_stop.command.
#[tauri::command]
pub async fn request_exit(app: AppHandle) {
    let app2 = app.clone();
    let yes = blocking(move || {
        app2.dialog()
            .message("การทำงานที่ค้างอยู่จะถูกหยุด และปิด Agent/UI backend; Model Server จะคงสถานะตาม Settings")
            .title("ออกจาก ENDEAVOR Agent?")
            .buttons(MessageDialogButtons::OkCancelCustom("ออกจากโปรแกรม".into(), "ยกเลิก".into()))
            .blocking_show()
    })
    .await;
    if yes {
        app.exit(0);
    }
}

/// A native yes/no question for the page (window.confirm is not usable there: see askConfirm in renderer.js).
#[tauri::command]
pub async fn confirm_dialog(app: AppHandle, message: String) -> Result<bool, ()> {
    Ok(blocking(move || {
        app.dialog()
            .message(message)
            .title("ยืนยัน")
            .buttons(MessageDialogButtons::OkCancelCustom("ตกลง".into(), "ยกเลิก".into()))
            .blocking_show()
    })
    .await)
}

/// The Model page: tools/models.py (the agent's own model_slots.py decides what is downloaded).
/// `check`: also ask Hugging Face for sizes not known yet (the page opening, or the Refresh button); progress polls
/// while a download runs pass false and read the disk only.
#[tauri::command]
pub async fn models_list(check: bool) -> Result<Value, ()> {
    Ok(blocking(move || crate::models::run(if check { &["list", "--check"] } else { &["list"] })).await)
}

#[tauri::command]
pub async fn model_action(action: String, model: String) -> Result<Value, ()> {
    if !["download", "cancel"].contains(&action.as_str()) {
        return Ok(json!({"ok": false, "error": format!("unknown action {action}")}));
    }
    Ok(blocking(move || crate::models::run(&[&action, &model])).await)
}

#[cfg(test)]
mod tests {
    #[test]
    fn numbers_print_like_javascript() {
        assert_eq!(super::fmt_number(1.0), "1");
        assert_eq!(super::fmt_number(0.9), "0.9");
        assert_eq!(super::fmt_number(1.2), "1.2");
    }

    #[test]
    fn only_listed_job_routes_pass() {
        assert!(super::JOB_ROUTES.contains(&("/tts/cancel", true)));
        assert!(super::JOB_ROUTES.contains(&("/tts/pause", true)));
        assert!(!super::JOB_ROUTES.contains(&("/tts/cancel", false)), "a control route is POST only");
        assert!(!super::JOB_ROUTES.contains(&("/runtime-settings", true)));
        assert!(!super::JOB_ROUTES.contains(&("/edit-access/delete", true)), "delete goes through the confirm dialog only");
    }

    #[test]
    fn voice_ids_are_header_safe() {
        assert_eq!(super::clean_voice("default"), "default");
        assert_eq!(super::clean_voice("a b"), "");
        assert_eq!(super::clean_voice("เสียง"), "");
        assert_eq!(super::clean_voice(&"x".repeat(65)), "");
    }
}
