pub mod client;
mod loaded;
mod manager;
mod scanner;
pub mod types;

pub const SUPPORTED_ABI_VERSION: u32 = 1;

pub use manager::PluginManager;
pub use scanner::PluginScanner;
pub use loaded::LoadedPlugin;