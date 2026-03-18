use serde_json::Value;

/// 当前任务的动态上下文。
///
/// 由调用方（UI 层 / API 层）在每次对话前构建，
/// 传递给 Orchestrator 用于决定本轮如何装配。
#[derive(Clone, Debug, Default)]
pub struct TaskContext {
    /// 任务类型（如 "creative_writing", "proofreading", "world_building"）
    pub task_type: String,

    /// 当前项目 ID（可选）
    pub project_id: Option<String>,

    /// 当前选区 / 焦点内容（如选中的文本段落）
    pub selection: Option<String>,

    /// 相关词条 / 实体（如角色名、地点名）
    pub entities: Vec<String>,

    /// 权限标记
    pub read_only: bool,

    /// 额外的键值对上下文（灵活扩展）
    pub extra: std::collections::HashMap<String, String>,
}

/// Orchestrator 的装配结果。
///
/// 包含本轮 LLM 调用所需的一切：
/// - 要注入的额外 system messages
/// - 筛选后的工具 schemas
/// - 参数覆盖
///
/// Session 消费这个结构，不需要知道它是怎么来的。
#[derive(Clone, Debug, Default)]
pub struct AssembledTurn {
    /// 额外注入的 system messages（在 Sense prompt 之后、用户消息之前）
    pub context_messages: Vec<String>,

    /// 本轮可用的工具 schemas（已筛选）
    pub tool_schemas: Option<Vec<Value>>,

    /// 本轮可用的工具名列表（用于执行时校验）
    pub enabled_tools: Vec<String>,

    /// 参数覆盖（优先级高于 Sense 默认值）
    pub model_override: Option<String>,
    pub temperature_override: Option<f64>,
    pub max_tokens_override: Option<i64>,
}