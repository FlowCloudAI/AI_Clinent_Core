use std::sync::Arc;
use tokio::sync::Mutex;
use crate::llm::tool::ToolFunctions;
use crate::llm::types::ChatRequest;

pub type SenseState<T> = Arc<Mutex<T>>;

pub fn sense_state_new<T: Default>() -> SenseState<T> {
    Arc::new(Mutex::new(T::default()))
}

#[allow(dead_code)]
pub trait SenseLoader {
    fn get_prompt(&self) -> Option<Vec<String>>;
    fn get_request(&self) -> Option<ChatRequest>;
    fn install_tool(&self, _tools: &mut ToolFunctions) -> anyhow::Result<String>;
}
