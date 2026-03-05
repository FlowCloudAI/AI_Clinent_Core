pub(crate) mod cup;
pub(crate) mod militech_acs;
pub(crate) mod llm_a;
pub(crate) mod llm_b;

#[allow(dead_code)]
pub struct Senses {
    pub cup: cup::CupSense,
    pub militech_acs: militech_acs::ACSSense,
    pub llm_a: llm_a::LLMASense,
    pub llm_b: llm_b::LLMBSense,
}

impl Senses {
    pub fn new() -> Self {
        Self {
            cup: cup::CupSense::new(),
            militech_acs: militech_acs::ACSSense::new(),
            llm_a: llm_a::LLMASense::new(),
            llm_b: llm_b::LLMBSense::new(),
        }
    }
}