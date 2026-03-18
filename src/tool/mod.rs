pub mod registry;
mod types;
mod executor;

// 重新导出辅助函数，保持现有调用方式兼容
pub use crate::llm::tool::{arg_i32, arg_str};