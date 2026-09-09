//! LLM 请求、响应与会话事件的公共数据结构。
//!
//! 供应商差异在这里归一化为稳定的核心语义；会话状态机只消费这些类型，
//! 不直接依赖任一厂商的 usage 字段形状。

use crate::error::ClientError;
use crate::llm::config::SecretString;
use crate::orchestrator::TaskContext;
use crate::plugin::types::ThinkingEffort;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::oneshot;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}
impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            reasoning_content: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            reasoning_content: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant(
        content: Option<impl Into<String>>,
        reasoning: Option<impl Into<String>>,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.map(|v| v.into()),
            reasoning_content: reasoning.map(|v| v.into()),
            tool_call_id: None,
            tool_calls,
        }
    }

    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            reasoning_content: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingType {
    #[serde(rename = "type")]
    pub thinking_type: String,
}

impl ThinkingType {
    pub fn enabled() -> ThinkingType {
        ThinkingType {
            thinking_type: "enabled".to_string(),
        }
    }

    pub fn disabled() -> ThinkingType {
        ThinkingType {
            thinking_type: "disabled".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,

    pub model: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingType>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_effort: Option<ThinkingEffort>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i32>,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: vec![],
            model: "".to_string(),
            thinking: None,
            thinking_effort: None,
            frequency_penalty: None,
            max_tokens: None,
            presence_penalty: None,
            response_format: None,
            stop: None,
            stream: None,
            stream_options: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: Some("auto".to_string()),
            logprobs: None,
            top_logprobs: None,
            n: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponseStream {
    pub id: String,
    pub object: String,
    pub choices: Vec<ChoiceStream>,
    pub created: i64,
    pub model: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: i64,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceStream {
    pub index: i64,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// 插件可将厂商返回的累计正文标记为快照，由流解码器转换为增量。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_snapshot: Option<String>,
    /// 插件可将厂商返回的累计思考内容标记为快照，由流解码器转换为增量。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content_snapshot: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    /// 本次请求从供应商缓存读取的输入 token；None 表示供应商未返回该口径。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_prompt_tokens: Option<i64>,
    /// 本次请求写入供应商缓存的输入 token；None 表示供应商未返回该口径。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_prompt_tokens: Option<i64>,
    /// 汇总中包含的实际 API 请求数。单次供应商响应固定为 1。
    #[serde(skip_serializing_if = "is_one_request")]
    pub request_count: u32,
    /// 明确返回缓存读取口径的请求数，用于区分零命中与统计缺失。
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cache_usage_known_requests: u32,
    /// 明确返回缓存写入口径的请求数。
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cache_creation_usage_known_requests: u32,
}

fn is_one_request(value: &u32) -> bool {
    *value == 1
}

fn is_zero_u32(value: &u32) -> bool {
    *value == 0
}

impl Default for Usage {
    fn default() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            cached_prompt_tokens: None,
            cache_creation_prompt_tokens: None,
            request_count: 1,
            cache_usage_known_requests: 0,
            cache_creation_usage_known_requests: 0,
        }
    }
}

impl<'de> Deserialize<'de> for Usage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let object = value
            .as_object()
            .ok_or_else(|| serde::de::Error::custom("usage 必须是 JSON 对象"))?;

        let direct_prompt = non_negative_i64(object.get("prompt_tokens"));
        let input_tokens = non_negative_i64(object.get("input_tokens"));
        let cache_read_input = non_negative_i64(object.get("cache_read_input_tokens"));
        let cache_creation_input = non_negative_i64(object.get("cache_creation_input_tokens"));
        let cache_hit = non_negative_i64(object.get("prompt_cache_hit_tokens"));
        let cache_miss = non_negative_i64(object.get("prompt_cache_miss_tokens"));

        let nested_cached = object
            .get("prompt_tokens_details")
            .and_then(Value::as_object)
            .and_then(|details| non_negative_i64(details.get("cached_tokens")))
            .or_else(|| {
                object
                    .get("input_tokens_details")
                    .and_then(Value::as_object)
                    .and_then(|details| non_negative_i64(details.get("cached_tokens")))
            });
        let cached_prompt_tokens = non_negative_i64(object.get("cached_prompt_tokens"))
            .or(cache_hit)
            .or(cache_read_input)
            .or(nested_cached);
        let cache_creation_prompt_tokens =
            non_negative_i64(object.get("cache_creation_prompt_tokens")).or(cache_creation_input);

        // OpenAI/DeepSeek 的 prompt_tokens 已包含缓存部分；Anthropic 风格的
        // input_tokens 则只表示未缓存输入，需要与读写缓存 token 相加。
        let prompt_tokens = direct_prompt
            .or_else(|| match (cache_hit, cache_miss) {
                (Some(hit), Some(miss)) => Some(hit.saturating_add(miss)),
                _ => None,
            })
            .or_else(|| {
                input_tokens.map(|input| {
                    input
                        .saturating_add(cache_read_input.unwrap_or(0))
                        .saturating_add(cache_creation_input.unwrap_or(0))
                })
            })
            .unwrap_or(0);
        let completion_tokens = non_negative_i64(object.get("completion_tokens"))
            .or_else(|| non_negative_i64(object.get("output_tokens")))
            .unwrap_or(0);
        let total_tokens = non_negative_i64(object.get("total_tokens"))
            .unwrap_or_else(|| prompt_tokens.saturating_add(completion_tokens));

        Ok(Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cached_prompt_tokens,
            cache_creation_prompt_tokens,
            request_count: 1,
            cache_usage_known_requests: u32::from(cached_prompt_tokens.is_some()),
            cache_creation_usage_known_requests: u32::from(cache_creation_prompt_tokens.is_some()),
        })
    }
}

fn non_negative_i64(value: Option<&Value>) -> Option<i64> {
    value.and_then(Value::as_i64).filter(|value| *value >= 0)
}

impl Usage {
    /// 合并同一次请求的累计/尾包统计，只取各字段的最新已知上界，绝不相加。
    pub(crate) fn merge_request_update(&mut self, update: Self) {
        self.prompt_tokens = self.prompt_tokens.max(update.prompt_tokens);
        self.completion_tokens = self.completion_tokens.max(update.completion_tokens);
        self.total_tokens = self.total_tokens.max(update.total_tokens);
        self.cached_prompt_tokens =
            max_known(self.cached_prompt_tokens, update.cached_prompt_tokens);
        self.cache_creation_prompt_tokens = max_known(
            self.cache_creation_prompt_tokens,
            update.cache_creation_prompt_tokens,
        );
        self.request_count = 1;
        self.cache_usage_known_requests = u32::from(self.cached_prompt_tokens.is_some());
        self.cache_creation_usage_known_requests =
            u32::from(self.cache_creation_prompt_tokens.is_some());
    }

    /// 将另一条实际请求的用量加入回合汇总，同时保留缓存统计覆盖率。
    pub(crate) fn accumulate_request(&mut self, request: &Self) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(request.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(request.completion_tokens);
        self.total_tokens = self.total_tokens.saturating_add(request.total_tokens);
        self.cached_prompt_tokens =
            sum_known(self.cached_prompt_tokens, request.cached_prompt_tokens);
        self.cache_creation_prompt_tokens = sum_known(
            self.cache_creation_prompt_tokens,
            request.cache_creation_prompt_tokens,
        );
        self.request_count = self.request_count.saturating_add(request.request_count);
        self.cache_usage_known_requests = self
            .cache_usage_known_requests
            .saturating_add(request.cache_usage_known_requests);
        self.cache_creation_usage_known_requests = self
            .cache_creation_usage_known_requests
            .saturating_add(request.cache_creation_usage_known_requests);
    }
}

fn max_known(current: Option<i64>, update: Option<i64>) -> Option<i64> {
    match (current, update) {
        (Some(current), Some(update)) => Some(current.max(update)),
        (Some(current), None) => Some(current),
        (None, Some(update)) => Some(update),
        (None, None) => None,
    }
}

fn sum_known(current: Option<i64>, request: Option<i64>) -> Option<i64> {
    match (current, request) {
        (Some(current), Some(request)) => Some(current.saturating_add(request)),
        (Some(current), None) => Some(current),
        (None, Some(request)) => Some(request),
        (None, None) => None,
    }
}

/// ---- tool call 结构（用于 a / stream delta 累积） ----
pub struct ToolFunctionArg {
    pub name: String,
    pub r#type: String,
    pub required: Option<bool>,
    pub description: Option<String>,
    pub default: Option<Value>,
    pub max: Option<Value>,
    pub min: Option<Value>,
    pub enum_values: Option<Vec<Value>>,
    pub items: Option<Box<Value>>,
    pub format: Option<String>,
    pub properties: Option<Value>,
    pub additional_properties: Option<Value>,
}

impl ToolFunctionArg {
    pub fn new(name: impl Into<String>, r#type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            r#type: r#type.into(),
            required: Some(false),
            description: None,
            default: None,
            max: None,
            min: None,
            enum_values: None,
            items: None,
            format: None,
            properties: None,
            additional_properties: None,
        }
    }

    pub fn required(mut self, required: bool) -> Self {
        self.required = Some(required);
        self
    }

    pub fn desc(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn default<V: Into<Value>>(mut self, default: V) -> Self {
        self.default = Some(default.into());
        self
    }

    pub fn max<V: Into<Value>>(mut self, max: V) -> Self {
        self.max = Some(max.into());
        self
    }

    pub fn min<V: Into<Value>>(mut self, min: V) -> Self {
        self.min = Some(min.into());
        self
    }

    pub fn enum_values<I, V>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: Into<Value>,
    {
        self.enum_values = Some(values.into_iter().map(Into::into).collect());
        self
    }

    pub fn items<V: Into<Value>>(mut self, items: V) -> Self {
        self.items = Some(Box::new(items.into()));
        self
    }

    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    pub fn properties<V: Into<Value>>(mut self, properties: V) -> Self {
        self.properties = Some(properties.into());
        self
    }

    pub fn additional_properties<V: Into<Value>>(mut self, additional_properties: V) -> Self {
        self.additional_properties = Some(additional_properties.into());
        self
    }

    pub fn schema(&self) -> Value {
        let mut v = serde_json::json!({"type": self.r#type});

        if let Some(desc) = &self.description {
            v["description"] = serde_json::json!(desc);
        }
        if let Some(vv) = &self.default {
            v["default"] = vv.clone();
        }
        if let Some(vv) = &self.max {
            v["maximum"] = vv.clone();
        }
        if let Some(vv) = &self.min {
            v["minimum"] = vv.clone();
        }
        if let Some(vv) = &self.enum_values {
            v["enum"] = serde_json::json!(vv);
        }
        if let Some(vv) = &self.items {
            v["items"] = (**vv).clone();
        }
        if let Some(vv) = &self.format {
            v["format"] = serde_json::json!(vv);
        }
        if let Some(vv) = &self.properties {
            v["properties"] = vv.clone();
        }
        if let Some(vv) = &self.additional_properties {
            v["additionalProperties"] = vv.clone();
        }
        v
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolFunctionCall {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default)]
    pub call_type: Option<String>,
    #[serde(default)]
    pub function: ToolFunctionCall,
    #[serde(default)]
    pub index: usize,
}

pub struct EventInfo {
    pub time_stamp: std::time::SystemTime,
    pub seq: u64,
    pub turn_id: u64,
}

/// 数据流事件
///
/// event_info: EventInfo 事件信息，
///
/// payload: DSEventPayload 事件负载，包含事件的具体内容。
pub struct DecoderEvent {
    pub event_info: EventInfo,
    pub payload: DecoderEventPayload,
}

/// 轮次结束状态
#[derive(Clone, Debug)]
pub enum TurnStatus {
    Ok,
    Cancelled,
    Interrupted,
    Error(ClientError),
}

pub enum DecoderEventPayload {
    TurnStart {
        model: String,
    },
    AssistantContentDelta {
        delta: String,
    },
    AssistantReasoningDelta {
        delta: String,
    },
    ToolCallStart {
        index: usize,
        tool_name: String,
    },
    ToolCallDelta {
        index: usize,
        tool_name: Option<String>,
        args: String,
    },
    ToolCallsRequired,
    TurnEnd {
        status: TurnStatus,
        finish_reason: Option<String>,
        usage: Option<Usage>,
    },
}

/// 会话控制指令（通过 SessionHandle 发往 drive loop）
#[derive(Debug)]
pub(crate) enum CtrlMsg {
    /// 切换到另一个插件（下一轮生效）
    SwitchPlugin {
        plugin_id: String,
        api_key: SecretString,
    },
    /// 将消息树 head 移动到指定节点（重说 / 分支 / 历史回退）
    Checkout { node_id: u64 },
    /// 从当前未完成的 assistant head 继续生成，不写入伪造的用户节点。
    Continue { node_id: u64 },
    /// 预检下一条用户消息的真实请求装配，不写入消息树或推进会话。
    Preflight {
        pending_user_message: String,
        response: oneshot::Sender<Result<RequestPreflight, ClientError>>,
    },
    /// 将用户正文与提交瞬间的上下文作为一个原子操作写入会话。
    SubmitUserTurn {
        message: String,
        context: Option<TaskContext>,
        response: oneshot::Sender<Result<u64, ClientError>>,
    },
}

/// 下一次有效请求的上下文预算预检结果。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestPreflight {
    pub estimated_input_tokens: u64,
    pub context_window_tokens: Option<u64>,
    pub output_reserve_tokens: Option<u64>,
    pub safety_reserve_tokens: Option<u64>,
    pub input_budget_tokens: Option<u64>,
    /// None 表示模型窗口未知，无法给出压缩建议。
    pub suggest_compaction: Option<bool>,
    pub head_node_id: Option<u64>,
    /// 对模型、工具、历史、上下文和待提交正文的确定性指纹。
    pub request_fingerprint: String,
    pub estimate_source: String,
}

#[derive(Debug, Clone)]
pub enum SessionEvent {
    NeedInput,
    TurnBegin {
        turn_id: u64,
        /// 本轮开始时的 head 节点 ID（用户消息节点，或工具结果节点）
        /// 树为空时为 0（实践中不会发生）
        node_id: u64,
    },
    ReasoningDelta(String),
    ContentDelta(String),

    ToolCall {
        index: usize,
        name: String,
        arguments: String,
    },
    ToolRetrying {
        index: usize,
        name: String,
        attempt: usize,
        max_retries: usize,
        delay_ms: u64,
    },
    ToolResult {
        index: usize,
        output: String,
        is_error: bool,
    },

    /// 发送副本因上下文预算被机械裁剪；持久化消息树保持不变。
    ContextTrimmed {
        dropped_rounds: usize,
        truncated_messages: usize,
        before: u64,
        after: u64,
        suggest_compaction: bool,
        estimate_source: String,
    },

    /// 一次实际 HTTP 请求已返回 usage；修复重试共享 request_id，以 attempt 区分。
    RequestUsage {
        /// 当前用户消息节点 ID；同一工具循环内的多次请求保持一致。
        turn_id: u64,
        request_id: u64,
        attempt: u32,
        usage: Usage,
    },

    TurnEnd {
        status: TurnStatus,
        /// 本轮助手消息节点 ID；没有产生任何助手输出时为 None。
        node_id: Option<u64>,
        /// 供应商返回的结束原因，例如 stop / length / tool_calls。
        finish_reason: Option<String>,
        /// 本轮若由续写触发，记录被续写的助手节点 ID。
        continuation_of: Option<u64>,
        /// API 用量统计（通常在流式最后一个 chunk 或非流式响应中返回）
        usage: Option<Usage>,
        /// 用本轮真实 prompt usage 更新后的 token 估算校准系数。
        calibration_factor: Option<f64>,
    },
    /// 分支切换完成（checkout 成功）。
    BranchChanged {
        node_id: u64,
    },
    Error(ClientError),
}

#[cfg(test)]
mod tests {
    use super::Usage;

    #[test]
    fn usage_兼容旧三字段响应且缓存口径未知() {
        let usage: Usage = serde_json::from_str(
            r#"{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}"#,
        )
        .unwrap();

        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
        assert_eq!(usage.cached_prompt_tokens, None);
        assert_eq!(usage.cache_usage_known_requests, 0);
        assert_eq!(usage.request_count, 1);
    }

    #[test]
    fn usage_归一化嵌套缓存字段且不重复计入输入总量() {
        let usage: Usage = serde_json::from_str(
            r#"{
                "prompt_tokens":100,
                "completion_tokens":20,
                "total_tokens":120,
                "prompt_tokens_details":{"cached_tokens":80}
            }"#,
        )
        .unwrap();

        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.cached_prompt_tokens, Some(80));
        assert_eq!(usage.total_tokens, 120);
        assert_eq!(usage.cache_usage_known_requests, 1);
    }

    #[test]
    fn usage_归一化命中未命中与_anthropic_缓存字段() {
        let deepseek: Usage = serde_json::from_str(
            r#"{
                "prompt_cache_hit_tokens":70,
                "prompt_cache_miss_tokens":30,
                "completion_tokens":5
            }"#,
        )
        .unwrap();
        assert_eq!(deepseek.prompt_tokens, 100);
        assert_eq!(deepseek.cached_prompt_tokens, Some(70));
        assert_eq!(deepseek.total_tokens, 105);

        let anthropic: Usage = serde_json::from_str(
            r#"{
                "input_tokens":10,
                "output_tokens":5,
                "cache_read_input_tokens":70,
                "cache_creation_input_tokens":20
            }"#,
        )
        .unwrap();
        assert_eq!(anthropic.prompt_tokens, 100);
        assert_eq!(anthropic.cached_prompt_tokens, Some(70));
        assert_eq!(anthropic.cache_creation_prompt_tokens, Some(20));
        assert_eq!(anthropic.total_tokens, 105);
    }

    #[test]
    fn usage_请求更新不累加而回合汇总保留覆盖率() {
        let mut request = Usage {
            prompt_tokens: 10,
            completion_tokens: 0,
            total_tokens: 10,
            cached_prompt_tokens: Some(4),
            cache_creation_prompt_tokens: None,
            request_count: 1,
            cache_usage_known_requests: 1,
            cache_creation_usage_known_requests: 0,
        };
        request.merge_request_update(Usage {
            prompt_tokens: 12,
            completion_tokens: 3,
            total_tokens: 15,
            cached_prompt_tokens: Some(5),
            ..Usage::default()
        });
        assert_eq!(request.total_tokens, 15);
        assert_eq!(request.cached_prompt_tokens, Some(5));

        let mut turn = request.clone();
        turn.accumulate_request(&Usage {
            prompt_tokens: 20,
            completion_tokens: 5,
            total_tokens: 25,
            ..Usage::default()
        });
        assert_eq!(turn.total_tokens, 40);
        assert_eq!(turn.request_count, 2);
        assert_eq!(turn.cached_prompt_tokens, Some(5));
        assert_eq!(turn.cache_usage_known_requests, 1);
    }
}
