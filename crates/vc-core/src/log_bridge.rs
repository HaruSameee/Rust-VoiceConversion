use std::sync::{Arc, OnceLock, RwLock};

type LogSink = dyn Fn(&str, &str) + Send + Sync + 'static;

fn sink_cell() -> &'static RwLock<Option<Arc<LogSink>>> {
    static CELL: OnceLock<RwLock<Option<Arc<LogSink>>>> = OnceLock::new();
    CELL.get_or_init(|| RwLock::new(None))
}

pub fn set_log_sink(sink: Option<Arc<LogSink>>) {
    if let Ok(mut guard) = sink_cell().write() {
        *guard = sink;
    }
}

pub fn emit_log(level: &str, message: &str) {
    let sink = sink_cell()
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned());
    if let Some(sink) = sink {
        sink(level, message);
    }
}
