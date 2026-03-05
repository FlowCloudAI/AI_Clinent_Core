use crate::http_poster::HttpPoster;
use crate::llm::stream_decoder::StreamDecoder;
use crate::llm::tool::ToolFunctions;
use crate::llm::types::{ChatRequest, ChatResponse, DecoderEventPayload, Message, SessionEvent, ThinkingType, ToolCall, ToolFunctionCall, TurnStatus};
use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;

#[derive(Default)]
struct ToolCallAccumulator {
    names: HashMap<usize, String>,
    args: HashMap<usize, String>,
    order: Vec<usize>,
    seen: std::collections::HashSet<usize>,
}

impl ToolCallAccumulator {
    fn touch(&mut self, index: usize) {
        if self.seen.insert(index) {
            self.order.push(index);
        }
        self.args.entry(index).or_default();
    }

    fn on_start(&mut self, index: usize, name: Option<&str>) {
        self.touch(index);
        if let Some(name) = name.filter(|s| !s.is_empty()) {
            self.names.insert(index, name.to_string());
        }
    }

    fn on_delta(&mut self, index: usize, name: Option<&str>, args_delta: &str) {
        self.touch(index);
        if let Some(name) = name.filter(|s| !s.is_empty()) {
            self.names.insert(index, name.to_string());
        }
        if !args_delta.is_empty() {
            self.args.get_mut(&index).unwrap().push_str(args_delta);
        }
    }

    fn build_calls(self, turn_id: u64) -> Vec<ToolCall> {
        let mut out = Vec::new();
        for index in self.order {
            let name = self.names.get(&index).cloned().unwrap_or_default();
            let mut args = self.args.get(&index).cloned().unwrap_or_default();

            if args.trim().is_empty() || serde_json::from_str::<Value>(&args).is_err() {
                args = "{}".to_string();
            }
            if name.is_empty() { continue; }

            out.push(ToolCall {
                id: Some(AIChatSession::synth_tool_call_id(turn_id, index)), // 本地合成
                call_type: Some("function".to_string()),
                function: ToolFunctionCall { name, arguments: args },
                index,
            });
        }
        out
    }
}

pub struct SessionHandle {
    inner: Arc<RwLock<ChatRequest>>,
}

impl SessionHandle {
    pub async fn get_conversation(&self) -> ChatRequest {
        self.inner.read().await.clone()
    }
}

pub struct AIChatSession {
    /// API client
    pub client: HttpPoster,

    /// 对话请求与历史（messages、tools、stream、thinking 等都在这里）
    pub conversation: ChatRequest,

    /// 工具函数管理器（schemas + conduct）
    pub tool: ToolFunctions,

    /// 流解码器（你现有的 decoder）
    pub decoder: StreamDecoder,

    /// 当前 turn id（纯粹用于标记事件）
    turn_id: u64,

    /// 事件 channel 的缓冲大小（大一点不容易背压卡住）
    pub event_buffer: usize,

    base_url: String,

    api_key: String,

    conv: Arc<RwLock<ChatRequest>>,
}

impl AIChatSession {
    pub fn new() -> Self {
        let client = HttpPoster::new();
        Self {
            client,
            conversation: ChatRequest::default(),
            tool: ToolFunctions::new(),
            decoder: Default::default(),
            turn_id: 0,
            event_buffer: 256,
            base_url: "".to_string(),
            api_key: "".to_string(),
            conv: Arc::new(Default::default()),
        }
    }

    #[allow(dead_code)]
    pub fn set_api(mut self, base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self.api_key = api_key.into();
        self
    }

    #[allow(dead_code)]
    pub async fn load_conversation(mut self, data: ChatRequest) -> Self {
        self.conversation = data;
        self
    }

    pub fn get_conversation(&self) -> ChatRequest {
        self.conversation.clone()
    }

    #[allow(dead_code)]
    pub async fn load_sense(mut self, sense: impl crate::llm::sense::SenseLoader) -> Result<Self> {
        match sense.get_request() {
            Some(request) => self.conversation = request,
            None => {}
        };
        match sense.get_prompt() {
            Some(prompts) => {
                for prompt in prompts {
                    self.add_message(Message::system(prompt)).await;
                }
            }
            None => {}
        };
        sense.install_tool(&mut self.tool)?;
        self.conversation.tools = self.tool.schemas();
        Ok(self)
    }

    #[allow(dead_code)]
    pub async fn load_json(mut self, address: &str) -> Result<Self> {
        let content = match tokio::fs::read_to_string(address).await {
            Ok(content) => content,
            Err(err) => {
                return Err(anyhow::anyhow!("Error reading file: {}", err));
            }
        };
        let data: ChatRequest = serde_json::from_str(content.as_str())?;
        self.conversation = data;
        Ok(self)
    }

    #[allow(dead_code)]
    pub fn set_model(mut self, model: impl Into<String>) -> Self {
        self.conversation.model = model.into();
        self
    }
    #[allow(dead_code)]
    pub fn set_temperature(mut self, temp: f64) -> Self {
        self.conversation.temperature = Some(temp);
        self
    }
    #[allow(dead_code)]
    pub fn set_stream(mut self, stream: bool) -> Self {
        self.conversation.stream = Some(stream);
        self
    }

    #[allow(dead_code)]
    pub fn set_max_tokens(mut self, max_tokens: i64) -> Self {
        self.conversation.max_tokens = Some(max_tokens);
        self
    }

    #[allow(dead_code)]
    pub fn set_top_p(mut self, top_p: f64) -> Self {
        self.conversation.top_p = Some(top_p);
        self
    }

    #[allow(dead_code)]
    pub fn set_presence_penalty(mut self, presence_penalty: f64) -> Self {
        self.conversation.presence_penalty = Some(presence_penalty);
        self
    }

    #[allow(dead_code)]
    pub fn set_frequency_penalty(mut self, frequency_penalty: f64) -> Self {
        self.conversation.frequency_penalty = Some(frequency_penalty);
        self
    }

    #[allow(dead_code)]
    pub fn set_thinking(mut self, thinking: bool) -> Self {
        if thinking {
            self.conversation.thinking = Some(ThinkingType::enabled());
        } else {
            self.conversation.thinking = Some(ThinkingType::disabled());
        }
        self
    }

    /// 对外暴露：启动会话“状态机”，返回一个 Stream<SessionEvent>
    ///
    /// - UI 层把用户输入通过 input_tx 发进来（input_rx 由 Session 持有）
    /// - UI 层消费这个 Stream 并渲染/交互
    ///
    /// 典型用法（UI 层）：
    ///   let (input_tx, input_rx) = mpsc::channel(16);
    ///   let stream = session.run(input_rx);
    ///   while let Some(ev) = stream.next().await { ... }
    pub fn run(
        mut self,
        mut input_rx: mpsc::Receiver<String>,
    ) -> (ReceiverStream<SessionEvent>, SessionHandle) {
        let (event_tx, event_rx) = mpsc::channel::<SessionEvent>(self.event_buffer);

        self.conv = Arc::new(RwLock::new(self.conversation.clone()));
        let handle = SessionHandle {
            inner: self.conv.clone(),
        };

        // 用 tokio 任务驱动 Session 主循环
        tokio::spawn(async move {
            // drive() 里出错时，用 Error 事件通知 UI
            if let Err(e) = self.drive(&mut input_rx, event_tx.clone()).await {
                let _ = event_tx.send(SessionEvent::Error(format!("{:#}", e))).await;
            }
        });

        (ReceiverStream::new(event_rx), handle)
    }

    async fn sync_conv(&self, conv: &Arc<RwLock<ChatRequest>>) {
        *conv.write().await = self.conversation.clone();
    }

    /// 会话主循环（核心状态机）
    ///
    /// 状态逻辑：
    /// 1) 若需要用户输入 -> emit NeedInput -> 等 input_rx.recv()
    /// 2) send_and_process -> 产出 reasoning/content/tool_calls/finish_reason
    /// 3) 写入 assistant 消息到 history
    /// 4) 若 finish_reason == tool_calls -> 执行工具 -> 继续下一轮（不需要用户输入）
    /// 5) 若 stop/其他 -> 回到 1)，继续等用户输入
    async fn drive(
        &mut self,
        input_rx: &mut mpsc::Receiver<String>,
        event_tx: mpsc::Sender<SessionEvent>,
    ) -> Result<()> {
        loop {
            // 决定是否需要用户输入：
            // - 这里用你原来的“最后一条 message 是否 assistant”规则（简单好用）
            // - 更高级可换成显式状态机枚举
            let need_user_input = self.should_wait_for_user();

            if need_user_input {
                // 通知 UI：我现在需要输入
                event_tx.send(SessionEvent::NeedInput).await.ok();

                // 等 UI 把输入发回来；如果 UI 关掉输入通道，直接结束会话
                let input = match input_rx.recv().await {
                    Some(s) => s,
                    None => {
                        // 输入端关闭：通常意味着 UI 结束
                        return Ok(());
                    }
                };

                // 记录用户消息
                self.add_message(Message::user(input)).await;
            }

            // 一轮开始
            self.turn_id += 1;
            event_tx
                .send(SessionEvent::TurnBegin {
                    turn_id: self.turn_id,
                })
                .await
                .ok();

            // 发请求并处理响应（流式/非流式都在里面）
            let (content, reasoning, tool_calls, finish_reason, turn_status) =
                self.send_and_process(&event_tx).await?;

            //println!("reasoning: {:?}", reasoning);

            // 写入 assistant 消息到历史
            self.add_message(Message::assistant(
                Some(content).filter(|s| !s.is_empty()),
                Some(reasoning).filter(|s| !s.is_empty()),
                tool_calls.clone(),
            )).await;

            // 关键：如果模型要求 tool_calls，则执行工具后立刻进入下一轮（不等用户输入）
            // ⚠️ 你原代码里 `!= "tool_calls"` 逻辑很可能反了，这里按常规语义写。
            if finish_reason.as_deref() == Some("tool_calls") {
                if let Some(calls) = tool_calls {
                    self.execute_tool_calls(calls, &event_tx).await?;
                    // 工具结果已经写回 history，继续下一轮
                    continue;
                }
            }

            // 一轮结束事件（让 UI 收个尾）
            event_tx
                .send(SessionEvent::TurnEnd {
                    status: turn_status.clone(),
                })
                .await
                .ok();

            // stop/cancelled 等情况：下一轮继续等用户输入（loop 顶部会判断）
        }
    }

    /// 根据历史最后一条消息判断是否需要等待用户输入
    fn should_wait_for_user(&self) -> bool {
        // 简单规则：如果历史为空或最后一条消息来自 assistant，则等待用户输入
        self.conversation
            .messages
            .last()
            .map_or(true, |msg| msg.role == "assistant")
    }

    /// 发送请求并处理响应：
    /// - 统一封装：流式/非流式
    /// - 处理过程中不断 emit SessionEvent 给 UI
    ///
    /// 返回：
    /// - content/reasoning：完整内容（用于写入历史）
    /// - tool_calls：完整工具调用（用于写入历史 + 执行）
    /// - finish_reason：stop/tool_calls/...
    /// - turn_status：本轮最终状态（Ok/Cancelled/Interrupted/Error）
    async fn send_and_process(
        &mut self,
        event_tx: &mpsc::Sender<SessionEvent>,
    ) -> Result<(
        String,
        String,
        Option<Vec<ToolCall>>,
        Option<String>,
        TurnStatus,
    )> {
        // clone 出请求，避免边处理边借用 self.conversation 导致的借用冲突
        let req = self.conversation.clone();

        if !req.stream.unwrap_or(false) {
            self.handle_non_stream(&req, event_tx).await
        } else {
            self.handle_stream(&req, event_tx).await
        }
    }

    /// 非流式响应处理：
    /// - 一次性拿到 reasoning/content/tool_calls
    /// - 但为了 UI 统一，仍然以 Delta 事件方式 emit（一次性 delta）
    async fn handle_non_stream(
        &mut self,
        req: &ChatRequest,
        event_tx: &mpsc::Sender<SessionEvent>,
    ) -> Result<(
        String,
        String,
        Option<Vec<ToolCall>>,
        Option<String>,
        TurnStatus,
    )> {

        let json = serde_json::to_value(req)?;

        // 获取 stream（你现有的 client API）
        let stream = self
            .client
            .post_json(self.base_url.clone(), self.api_key.clone(), json)
            .await
            .context("创建流式请求失败")?;
        tokio::pin!(stream);

        let raw_line = match stream.next().await {
            Some(line) => line,
            None => return Err(anyhow!("stream closed unexpectedly")),
        };

        let line = match raw_line {
            Ok(line) => line,
            Err(err) => return Err(anyhow!("stream error: {}", err))
        };

        let res: ChatResponse = serde_json::from_str(&line)?;

        // 这里假设 choices 至少一个
        let choice = &res.choices[0];

        let reasoning = choice.message.reasoning_content.clone().unwrap_or_default();
        let content = choice.message.content.clone().unwrap_or_default();

        // 工具调用（非流式也可能有）
        let tool_calls_vec = choice.message.tool_calls.clone().unwrap_or_default();
        let tool_calls = if tool_calls_vec.is_empty() {
            None
        } else {
            // emit 每个 tool_call（UI 可记录/展示）
            for call in &tool_calls_vec {
                call
                    .id
                    .clone()
                    .unwrap_or_else(|| "unknown_tool_call_id".to_string());
                let name = call.function.name.clone();
                let index = call.index;
                event_tx
                    .send(SessionEvent::ToolCall {
                        index,
                        name,
                    })
                    .await
                    .ok();
            }
            Some(tool_calls_vec)
        };

        // emit reasoning/content（一次性 delta）
        if !reasoning.is_empty() {
            event_tx
                .send(SessionEvent::ReasoningDelta(reasoning.clone()))
                .await
                .ok();
        }
        if !content.is_empty() {
            event_tx
                .send(SessionEvent::ContentDelta(content.clone()))
                .await
                .ok();
        }

        // finish_reason（可能是 stop/tool_calls/…）
        let finish_reason = choice.finish_reason.clone();

        // 非流式一般认为成功就 Ok；如果你 res 里有更细状态也可映射
        let status = TurnStatus::Ok;

        Ok((content, reasoning, tool_calls, Some(finish_reason), status))
    }

    /// 流式响应处理：
    /// - 逐行读取 SSE/lines blob
    /// - 交给 decoder 解码出事件
    /// - 每个事件 emit 给 UI，同时累积 full_content/full_reasoning/tool_calls
    async fn handle_stream(
        &mut self,
        req: &ChatRequest,
        event_tx: &mpsc::Sender<SessionEvent>,
    ) -> Result<(
        String,
        String,
        Option<Vec<ToolCall>>,
        Option<String>,
        TurnStatus,
    )> {
        // decoder 开始新一轮（你的 decoder 需要 turn_id）
        self.decoder.begin_turn(self.turn_id);

        let mut full_content = String::new();
        let mut full_reasoning = String::new();
        let mut finish_reason: Option<String> = None;

        // 这里存“完整工具调用”（只在 ToolCallStart 时 take_ready）
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        // 轮状态：默认 Ok，遇到错误/取消/中断时更新
        let mut turn_status = TurnStatus::Ok;

        let mut acc = ToolCallAccumulator::default();

        let json = serde_json::to_value(req)?;

        // 获取 stream（你现有的 client API）
        let stream = self
            .client
            .post_json(self.base_url.clone(), self.api_key.clone(), json)
            .await
            .context("创建流式请求失败")?;
        tokio::pin!(stream);

        // 逐个 blob 处理
        'outer: while let Some(raw_line) = stream.next().await {
            let line = &*raw_line?;

            if line.is_empty() {
                continue;
            }

            // 把每行喂给 decoder，decoder 可能吐出多个事件
            let events = self.decoder.decode(line);

            for ev in events {
                let ev = ev?; // decoder 事件级错误直接上抛

                match ev.payload {
                    // 推理 delta
                    DecoderEventPayload::AssistantReasoningDelta { delta } => {
                        full_reasoning.push_str(&delta);
                        event_tx
                            .send(SessionEvent::ReasoningDelta(delta))
                            .await
                            .ok();
                    }

                    // 内容 delta
                    DecoderEventPayload::AssistantContentDelta { delta } => {
                        full_content.push_str(&delta);
                        event_tx.send(SessionEvent::ContentDelta(delta)).await.ok();
                    }

                    // 工具开始
                    DecoderEventPayload::ToolCallStart { index, tool_name } => {
                        acc.on_start(index, Some(&tool_name));
                        event_tx.send(SessionEvent::ToolCall { index, name: tool_name }).await.ok();
                    }

                    // 2) Delta：拼接 args
                    DecoderEventPayload::ToolCallDelta { index, tool_name, args } => {
                        acc.on_delta(index, tool_name.as_deref(), &args);
                    }

                    // 3) 关键：finish_reason == tool_calls 的“收口信号”
                    DecoderEventPayload::ToolCallsRequired => {
                        let calls = acc.build_calls(self.turn_id);
                        tool_calls.extend(calls);
                        finish_reason = Some("tool_calls".to_string());

                        for call in &tool_calls {
                            event_tx.send(SessionEvent::ToolCall {
                                index: call.index,
                                name: call.function.name.clone(),
                            }).await.ok();
                        }
                        break 'outer;
                    }

                    // 回合结束：对应 stop/cancelled/interrupted/error
                    DecoderEventPayload::TurnEnd { status, .. } => {
                        turn_status = status.clone();
                        match &turn_status {
                            TurnStatus::Ok => finish_reason = Some("stop".to_string()),
                            TurnStatus::Cancelled => {
                                finish_reason = Some("cancelled".to_string())
                            }
                            TurnStatus::Interrupted => {
                                finish_reason = Some("interrupted".to_string())
                            }
                            TurnStatus::Error(e) => {
                                return Err(anyhow::anyhow!(e.clone()));
                            }
                        }
                        break 'outer;
                    }

                    // 其他 payload 不关心
                    _ => {}
                }
            }
        }


        let tool_calls_opt = if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        };

        Ok((
            full_content,
            full_reasoning,
            tool_calls_opt,
            finish_reason,
            turn_status,
        ))
    }

    pub fn get_current_conversation(&self) -> ChatRequest {
        self.conversation.clone()
    }

    /// 执行工具调用并写回工具结果消息：
    /// - 对每个工具调用：
    ///   1) 解析参数 JSON（解析失败则 {}）
    ///   2) tool.conduct 执行
    ///   3) emit ToolResult 事件
    ///   4) add_message(Message::tool(...)) 写回历史
    async fn execute_tool_calls(
        &mut self,
        tool_calls: Vec<ToolCall>,
        event_tx: &mpsc::Sender<SessionEvent>,
    ) -> Result<()> {
        for call in tool_calls {
            let func_name = call.function.name.clone();

            let args_str = call.function.arguments.trim();

            let args_v: Value = match args_str.is_empty() {
                true => Value::Object(Default::default()),
                false => serde_json::from_str(args_str)?,
            };

            // 执行工具
            let output_res = self.tool.conduct(&func_name, Some(&args_v)).await;

            // 组装结果文本（你也可改成更结构化的 tool message）
            let (output, is_error) = match output_res {
                Ok(output) => (format!("工具执行成功: {}", output), false),
                Err(e) => (format!("工具执行失败: {}", e), true),
            };

            let index = call.index;
            let tool_call_id = AIChatSession::synth_tool_call_id(self.turn_id, index);

            // emit 给 UI
            event_tx
                .send(SessionEvent::ToolResult {
                    index,
                    output: output.clone(),
                    is_error,
                })
                .await
                .ok();

            // 写回历史（让下一轮模型能看到工具返回）
            self.add_message(Message::tool(output, tool_call_id)).await;
        }

        Ok(())
    }

    async fn add_message(&mut self, msg: Message) {
        self.conversation.messages.push(msg);
        self.sync_conv(&self.conv).await;
    }

    #[inline]
    fn synth_tool_call_id(turn_id: u64, index: usize) -> String {
        format!("t{}:idx:{}", turn_id, index)
    }
}