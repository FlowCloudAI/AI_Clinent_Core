use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use wasmtime::component::ResourceTable;
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[derive(PartialEq)]
pub enum PluginKind {
    #[serde(rename = "kind/llm")]
    LLM,
    #[serde(rename = "kind/image")]
    Image,
    #[serde(rename = "kind/tts")]
    TTS,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct PluginInfo {
    pub id: String,
    pub version: String,
    pub author: String,
    pub abi_version: u32,
    pub name: String,
    pub description: String,
    pub kind: PluginKind,
}

pub struct PluginMeta {
    pub id: String,
    pub name: String,
    pub description: String,
    pub author: String,
    pub version: String,
    pub kind: PluginKind,

    pub fcplug_path: PathBuf,
}

pub struct HostState {
    pub table: ResourceTable,
    pub wasi: WasiCtx,
}

impl WasiView for HostState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

pub(crate) mod plugin_bindings {
    wasmtime::component::bindgen!({
        path: "wit/plugin.wit",
        world: "api",
    });
}