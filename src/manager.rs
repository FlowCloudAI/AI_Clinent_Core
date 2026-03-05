use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use wasmtime::component::Linker;
use wasmtime::{Config, Engine};

use crate::loaded::LoadedPlugin;
use crate::scanner::PluginScanner;
use crate::types::{HostState, PluginKind, PluginMeta};
use crate::SUPPORTED_ABI_VERSION;

pub struct PluginManager {
    plug_path: PathBuf,
    pub plugins: HashMap<String, PluginMeta>,
    engine: Engine,
    linker: Linker<HostState>,

    // 运行态插件
    llm_plugin: LoadedPlugin,
    image_plugin: LoadedPlugin,
    tts_plugin: LoadedPlugin,
}

impl PluginManager {
    pub fn new(plug_path: PathBuf) -> Result<Self> {
        let mut config = Config::new();
        config.wasm_component_model(true);
        let engine = Engine::new(&config)?;
        let mut linker = Linker::new(&engine);

        wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;


        let plugins = PluginManager::load_plugins(Path::new(&plug_path))?;

        Ok(PluginManager {
            plug_path,
            plugins,
            engine,
            linker,
            llm_plugin: LoadedPlugin::new(PluginKind::LLM),
            image_plugin: LoadedPlugin::new(PluginKind::Image),
            tts_plugin: LoadedPlugin::new(PluginKind::TTS),
        })
    }

    fn load_plugins(path: &Path) -> Result<HashMap<String, PluginMeta>> {
        let mut plugins: HashMap<String, PluginMeta> = HashMap::new();

        for fcplug in PluginScanner::scan_plugins(path)? {
            match PluginScanner::read_plugin_info(&fcplug) {

                Ok(info) => {

                    if plugins.contains_key(&info.id) {
                        println!("duplicate plugin id {}", info.id);
                        continue;
                    }

                    if info.abi_version != SUPPORTED_ABI_VERSION {
                        println!("skip plugin {}: version mismatch", info.id);
                        continue;
                    }
                    plugins.insert(
                        info.id.clone(),
                        PluginScanner::build_plugin_meta(info, fcplug)
                    );
                }
                Err(e) => {
                    println!("invalid plugin {:?}: {}", fcplug, e);
                }
            }
        }

        Ok(plugins)
    }

    pub fn add_plugin(&mut self, plugin_path: String) -> Result<()> {

        let info = PluginScanner::read_plugin_info((&plugin_path).as_ref())?;

        if self.plugins.contains_key(&info.id) {
            return Err(anyhow!("plugin {} already exists", info.id))
        };

        if info.abi_version != SUPPORTED_ABI_VERSION {
            return Err(anyhow!("skip plugin {}: version mismatch", info.id))
        };

        let filename = Path::new(&plugin_path)
            .file_name()
            .ok_or_else(|| anyhow!("invalid plugin filename"))?;

        let dst = Path::new(&self.plug_path).join(filename);

        fs::copy(&plugin_path, &dst)
            .map_err(|e| anyhow!("copy plugin {} failed: {}", info.id, e))?;

        let fcplug = dst;

        self.plugins.insert(
            info.id.clone(),
            PluginScanner::build_plugin_meta(info, fcplug)
        );

        Ok(())
    }

    pub fn load_llm_plugin(&mut self, id: &str) -> Result<()> {
        self.llm_plugin.load(&self.plugins, &self.linker, &self.engine, id)?;
        Ok(())
    }

    pub fn load_image_plugin(&mut self, id: &str) -> Result<()> {
        self.image_plugin.load(&self.plugins, &self.linker, &self.engine, id)?;
        Ok(())
    }

    pub fn load_tts_plugin(&mut self, id: &str) -> Result<()> {
        self.tts_plugin.load(&self.plugins, &self.linker, &self.engine, id)?;
        Ok(())
    }

    pub fn llm_map_request(&mut self, req: &str) -> Result<String> {
        self.llm_plugin.map_request(req)
    }

    pub fn llm_map_response(&mut self, resp: &str) -> Result<String> {
        self.llm_plugin.map_response(resp)
    }

    pub fn image_map_request(&mut self, req: &str) -> Result<String> {
        self.image_plugin.map_request(req)
    }

    pub fn image_map_response(&mut self, resp: &str) -> Result<String> {
        self.image_plugin.map_response(resp)
    }

    pub fn tts_map_request(&mut self, req: &str) -> Result<String> {
        self.tts_plugin.map_request(req)
    }

    pub fn tts_map_response(&mut self, resp: &str) -> Result<String> {
        self.tts_plugin.map_response(resp)
    }
}
