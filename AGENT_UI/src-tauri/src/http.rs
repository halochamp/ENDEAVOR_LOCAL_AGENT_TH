//! The agent's REST side (main.js `_agentJsonRequest`): every request carries X-Auth-Token, answers are JSON capped at
//! 1 MB with `http_status` added, uploads stream from disk (a 250 MB document never sits in RAM).

use crate::agent::AGENT_PORT;
use serde_json::{json, Value};
use std::path::Path;
use std::time::Duration;

const MAX_ANSWER: usize = 1024 * 1024;

pub struct Upload<'a> {
    pub path: &'a Path,
    pub content_type: &'a str,
    pub headers: Vec<(&'static str, String)>,
}

pub async fn request(token: &str, method: &str, route: &str, json_body: Option<&Value>, upload: Option<Upload<'_>>, timeout: Duration) -> Value {
    let url = format!("http://127.0.0.1:{AGENT_PORT}{route}");
    let client = match reqwest::Client::builder().timeout(timeout).build() {
        Ok(c) => c,
        Err(e) => return json!({"ok": false, "error": e.to_string()}),
    };
    let mut req = if method == "POST" { client.post(&url) } else { client.get(&url) };
    req = req.header("X-Auth-Token", token);
    if let Some(body) = json_body {
        req = req.header("Content-Type", "application/json").body(body.to_string());
    }
    if let Some(up) = upload {
        let file = match tokio::fs::File::open(up.path).await {
            Ok(f) => f,
            Err(e) => return json!({"ok": false, "error": e.to_string()}),
        };
        let len = file.metadata().await.map(|m| m.len()).unwrap_or(0);
        req = req.header("Content-Type", up.content_type).header("Content-Length", len.to_string());
        for (k, v) in up.headers {
            req = req.header(k, v);
        }
        req = req.body(reqwest::Body::from(file));
    }
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            let msg = if e.is_timeout() { "Agent request timed out".to_string() } else { e.to_string() };
            return json!({"ok": false, "error": msg});
        }
    };
    let status = resp.status().as_u16();
    let mut body = Vec::new();
    let mut resp = resp;
    loop {
        match resp.chunk().await {
            Ok(Some(chunk)) => {
                body.extend_from_slice(&chunk);
                if body.len() > MAX_ANSWER {
                    return json!({"ok": false, "error": "agent response too large", "http_status": status});
                }
            }
            Ok(None) => break,
            Err(e) => return json!({"ok": false, "error": e.to_string(), "http_status": status}),
        }
    }
    let text = String::from_utf8_lossy(&body);
    match serde_json::from_str::<Value>(if text.trim().is_empty() { "{}" } else { &text }) {
        Ok(Value::Object(mut map)) => {
            map.insert("http_status".into(), json!(status));
            Value::Object(map)
        }
        _ => json!({"ok": false, "error": "invalid agent response", "http_status": status}),
    }
}

/// `encodeURIComponent` (the routes and file-name headers main.js built with it).
pub fn encode_component(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.!~*'()".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn encodes_like_encode_uri_component() {
        assert_eq!(super::encode_component("รายงาน ปี 2569.pdf"), "%E0%B8%A3%E0%B8%B2%E0%B8%A2%E0%B8%87%E0%B8%B2%E0%B8%99%20%E0%B8%9B%E0%B8%B5%202569.pdf");
        assert_eq!(super::encode_component("a&b=c/d"), "a%26b%3Dc%2Fd");
    }
}
