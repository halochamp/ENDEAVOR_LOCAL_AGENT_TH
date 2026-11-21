//! Agent TH's desktop UI on Tauri 2. The Rust core replaces the old Electron main process (agent lifecycle, dialogs, window
//! modes); the page is the proven Agent TH renderer in the new light layout.

mod agent;
mod commands;
mod http;
mod models;

use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_opener::OpenerExt;

pub const DEFAULT_W: f64 = 1280.0;
pub const DEFAULT_H: f64 = 820.0;
pub const MIN_W: f64 = 900.0;
pub const MIN_H: f64 = 600.0;

/// The page may only ever show the app itself; any other navigation goes to the browser (web links) or nowhere.
fn is_app_url(url: &tauri::Url) -> bool {
    matches!(url.scheme(), "tauri") || url.host_str() == Some("tauri.localhost")
}

/// One running copy (main.js `killOtherAppInstances`): a second copy would kill :8765 under the first one's chat.
/// The lock is an flock on ~/.agent_th_ui/app.lock, released by the OS when this process ends (even on a crash).
fn single_instance() -> bool {
    use std::os::unix::io::AsRawFd;
    let dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".agent_th_ui");
    let _ = std::fs::create_dir_all(&dir);
    let Ok(file) = std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(dir.join("app.lock")) else {
        return true; // cannot lock: do not refuse to start over it
    };
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return false;
    }
    std::mem::forget(file); // held for the life of the process
    true
}

pub fn run() {
    if !single_instance() {
        eprintln!("Agent TH is already running");
        return;
    }
    let agent = agent::Agent::new();
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(agent.clone())
        .manage(commands::WindowMode::default())
        .invoke_handler(tauri::generate_handler![
            commands::startup_state,
            commands::get_token,
            commands::open_workspace,
            commands::open_url,
            commands::set_runtime_settings,
            commands::agent_job,
            commands::agent_get,
            commands::pdf_to_text_start,
            commands::tts_start,
            commands::translate_start,
            commands::edit_file,
            commands::delete_file,
            commands::show_open_dialog,
            commands::show_pin_dialog,
            commands::show_approved_edit_folder_dialog,
            commands::set_compact,
            commands::request_exit,
            commands::confirm_dialog,
            commands::models_list,
            commands::model_action,
        ])
        .setup(move |app| {
            let handle = app.handle().clone();
            let opener = app.handle().clone();
            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("ENDEAVOR LOCAL AGENT TH")
                .inner_size(DEFAULT_W, DEFAULT_H)
                .min_inner_size(MIN_W, MIN_H)
                .title_bar_style(tauri::TitleBarStyle::Overlay)
                .hidden_title(true)
                .on_navigation(move |url| {
                    if is_app_url(url) {
                        return true;
                    }
                    if matches!(url.scheme(), "http" | "https") {
                        let _ = opener.opener().open_url(url.as_str(), None::<&str>);
                    }
                    false
                })
                .build()?;
            agent.startup(handle);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Agent TH");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            handle.state::<std::sync::Arc<agent::Agent>>().shutdown();
        }
    });
}
