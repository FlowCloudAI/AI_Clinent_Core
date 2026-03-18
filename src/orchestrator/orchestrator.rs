use anyhow::Result;
use std::sync::Arc;

use crate::orchestrator::context::{AssembledTurn, TaskContext};
use crate::sense::Sense;
use crate::tool::registry::ToolRegistry;

/// 任务编排器。
///
/// 输入：Sense + ToolRegistry + TaskContext
/// 输出：AssembledTurn（拼好的 prompt + tools + params）
///
/// 每次对话轮次开始前调用 `assemble`，结果交给 Session 消费。
pub struct TaskOrchestrator {
    /// 当前模式预设
    sense: Box<dyn Sense>,

    /// 全局工具库（共享引用）
    registry: Arc<ToolRegistry>,
}

impl TaskOrchestrator {
    pub fn new(sense: Box<dyn Sense>, registry: Arc<ToolRegistry>) -> Self {
        Self { sense, registry }
    }

    /// 替换当前 Sense（切换模式）。
    pub fn set_sense(&mut self, sense: Box<dyn Sense>) {
        self.sense = sense;
    }

    /// 获取当前 Sense 的系统提示词。
    pub fn prompts(&self) -> Vec<String> {
        self.sense.prompts()
    }

    /// 获取当前 Sense 的默认 ChatRequest。
    pub fn default_request(&self) -> Option<crate::llm::types::ChatRequest> {
        self.sense.default_request()
    }

    /// 核心方法：根据 Sense + ToolRegistry + TaskContext 装配本轮配置。
    ///
    /// 做三件事：
    /// 1. 从 TaskContext 选择要注入的上下文片段拼进 prompt
    /// 2. 根据 Sense 白名单从 ToolRegistry 筛选本轮可用工具
    /// 3. 根据任务类型决定参数覆盖
    pub fn assemble(&self, ctx: &TaskContext) -> Result<AssembledTurn> {
        let mut turn = AssembledTurn::default();

        // ── 1. 上下文注入 ──
        self.inject_context(ctx, &mut turn);

        // ── 2. 工具筛选 ──
        self.select_tools(&mut turn);

        // ── 3. 参数覆盖 ──
        self.apply_overrides(ctx, &mut turn);

        Ok(turn)
    }

    /// 获取 ToolRegistry 引用（Session 执行工具时需要）。
    pub fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }

    // ── 内部方法 ──

    /// 根据 TaskContext 生成额外的 system messages。
    fn inject_context(&self, ctx: &TaskContext, turn: &mut AssembledTurn) {
        // 注入任务类型
        if !ctx.task_type.is_empty() {
            turn.context_messages.push(
                format!("[Task type: {}]", ctx.task_type),
            );
        }

        // 注入选区内容
        if let Some(ref sel) = ctx.selection {
            turn.context_messages.push(
                format!("[Current selection]\n{}", sel),
            );
        }

        // 注入相关实体
        if !ctx.entities.is_empty() {
            turn.context_messages.push(
                format!("[Related entities: {}]", ctx.entities.join(", ")),
            );
        }

        // 注入额外上下文
        for (k, v) in &ctx.extra {
            turn.context_messages.push(
                format!("[{}]\n{}", k, v),
            );
        }
    }

    /// 根据 Sense 白名单筛选工具。
    fn select_tools(&self, turn: &mut AssembledTurn) {
        match self.sense.tool_whitelist() {
            Some(whitelist) => {
                // Sense 指定了白名单，只启用这些工具
                turn.enabled_tools = whitelist
                    .iter()
                    .filter(|name| self.registry.has_tool(name))
                    .cloned()
                    .collect();
                turn.tool_schemas = self.registry.schemas_filtered(&turn.enabled_tools);
            }
            None => {
                // 无白名单，启用所有工具
                turn.enabled_tools = self.registry.tool_names();
                turn.tool_schemas = self.registry.schemas();
            }
        }
    }

    /// 根据任务类型覆盖参数。
    ///
    /// 这里是策略层：不同任务类型有不同的最佳参数。
    /// 可以后续扩展为可配置的规则引擎。
    fn apply_overrides(&self, ctx: &TaskContext, turn: &mut AssembledTurn) {
        match ctx.task_type.as_str() {
            "creative_writing" => {
                turn.temperature_override = Some(0.85);
            }
            "proofreading" | "translation" => {
                turn.temperature_override = Some(0.1);
            }
            "code_generation" => {
                turn.temperature_override = Some(0.0);
            }
            _ => {}
        }
    }
}