mod senses;
mod apis;

use crate::senses::Senses;
use anyhow::Result;
use flowcloudai_client::llm::session::AIChatSession;
use flowcloudai_client::llm::types::{SessionEvent, TurnStatus};
use futures_util::StreamExt;
use std::io::{Write, stdout};
use tokio::sync::mpsc;

const SYSTEM_COLOR: &str = "\x1b[94m";
const REASONING_COLOR: &str = "\x1b[92m";
const TOOL_COLOR: &str = "\x1b[90m";
const TOOL_CALL_COLOR: &str = "\x1b[96m";
const COLOR_RESET: &str = "\x1b[0m";

#[derive(Default)]
struct BotState {
    // 本轮累计的 assistant content
    content_buf: String,
    // 等待喂给这个 bot 的输入（NeedInput 时发送）
    pending_input: Option<String>,
    // turn 计数（你也可以用 TurnBegin 的 turn_id）
    turns_finished: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Active {
    A,
    B,
}

#[tokio::main]
async fn main() -> Result<()> {
    let senses: Senses = Senses::new();

    // ====== Bot A ======
    let bot_a = AIChatSession::new()
        .set_api(
            apis::QWEN_LLM.url,
            apis::QWEN_LLM.key
        )
        .load_sense(senses.llm_b)
        .await?
        .set_model("qwen3.5-flash")
        .set_thinking(false)
        .set_frequency_penalty(0.0)
        .set_stream(true);

    // ====== Bot B ======
    let bot_b = AIChatSession::new()
        .set_api(
            apis::QWEN_LLM.url,
            apis::QWEN_LLM.key
        )
        .load_sense(senses.llm_a)
        .await?
        .set_model("qwen3.5-flash")
        .set_thinking(false)
        .set_stream(true);

    let (a_tx, a_rx) = mpsc::channel::<String>(32);
    let (b_tx, b_rx) = mpsc::channel::<String>(32);

    let (mut a_stream, _a_handle) = bot_a.run(a_rx);
    let (mut b_stream, _b_handle) = bot_b.run(b_rx);

    // 两个 bot 的运行状态
    let mut a_state = BotState::default();
    let mut b_state = BotState::default();

    // 限制对话轮数，避免无限聊天烧 token
    let max_turns = 10;

    let mut active = Active::A;

    let mut is_reasoning = false;

    loop {
        // 任意一边达到 max_turns 就停
        if a_state.turns_finished >= max_turns || b_state.turns_finished >= max_turns {
            println!(
                "\n{}Reached max_turns={}, stop.{}",
                SYSTEM_COLOR, max_turns, COLOR_RESET
            );
            break;
        }

        loop {
            tokio::select! {
                biased;

                ev = a_stream.next(), if active == Active::A => {
                    match ev {
                        None => break,
                        Some(ev) => {
                            handle_event(
                                Active::A,
                                ev,
                                &a_tx, &mut a_state,
                                &b_tx, &mut b_state,
                                &mut active,
                                &mut is_reasoning
                            ).await?;
                        }
                    }
                }

                ev = b_stream.next(), if active == Active::B => {
                    match ev {
                        None => break,
                        Some(ev) => {
                            handle_event(
                                Active::B,
                                ev,
                                &b_tx, &mut b_state,
                                &a_tx, &mut a_state,
                                &mut active,
                                &mut is_reasoning
                            ).await?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

///
/// who: 当前 bot 标签（"A" or "B"）
/// self_state: 当前 bot 的状态
/// self_tx: 当前 bot 的输入 sender（NeedInput 时用）
/// other_state: 对方 bot 的状态（用于塞 pending_input）
/// other_tx: 对方 bot 的输入 sender（NeedInput 时用）
///
async fn handle_event(
    me: Active,
    ev: SessionEvent,
    me_tx: &mpsc::Sender<String>,
    me_state: &mut BotState,
    _other_tx: &mpsc::Sender<String>,
    other_state: &mut BotState,
    active: &mut Active,
    is_reasoning: &mut bool,
) -> Result<()> {
    match ev {
        SessionEvent::NeedInput => {
            if let Some(msg) = me_state.pending_input.take() {
                let _ = me_tx.send(msg).await;
            }
        }

        SessionEvent::ContentDelta(delta) => {
            if *is_reasoning {
                println!();
            }
            *is_reasoning = false;
            // 只累积 content（你要把 reasoning 也转发就另加 buf）
            me_state.content_buf.push_str(&delta);
            print!("{}", delta);
            Write::flush(&mut stdout()).ok();
        }

        SessionEvent::ReasoningDelta(delta) => {
            *is_reasoning = true;
            print!("{}{}{}", REASONING_COLOR, delta, COLOR_RESET);
            Write::flush(&mut stdout()).ok();
        }

        SessionEvent::TurnEnd { status } => {
            println!(
                "\n{}--- {:?} TurnEnd: {:?} ---{}",
                SYSTEM_COLOR, me, status, COLOR_RESET
            );

            if matches!(status, TurnStatus::Ok) {
                // ✅ 把我这一轮说的话喂给对方
                let msg = me_state.content_buf.trim().to_string();
                me_state.content_buf.clear();

                if !msg.is_empty() {
                    other_state.pending_input = Some(msg);
                    // 可选：也可以直接 send，让对方不用等 NeedInput
                    // let _ = other_tx.send(msg).await;
                }

                // ✅ 关键：切换 active，让对方开始被 poll
                *active = match *active {
                    Active::A => Active::B,
                    Active::B => Active::A,
                };
            } else {
                // cancelled/interrupted/error 你可以选择 break 由外层处理
            }
        }

        SessionEvent::ToolCall { index, name } => {
            println!(
                "\n{}[{:?} ToolCall] index={} name={}{}",
                TOOL_CALL_COLOR, me, index, name, COLOR_RESET
            );
        }

        SessionEvent::ToolResult {
            index,
            output,
            is_error,
        } => {
            println!(
                "\n{}[{:?} ToolResult] index={} err={} output={}{}",
                TOOL_COLOR, me, index, is_error, output, COLOR_RESET
            );
        }

        SessionEvent::TurnBegin { turn_id } => {
            *is_reasoning = false;
            println!(
                "\n{}=== {:?} Turn {} ==={}",
                SYSTEM_COLOR, me, turn_id, COLOR_RESET
            );
        }

        SessionEvent::Error(msg) => {
            println!("\n[{:?} Error] {}", me, msg);
        }
    }

    Ok(())
}
