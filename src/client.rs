use anyhow::Result;
use reqwest::Client;
use serde_json::Value;

pub struct HttpClient {
    client: Client,
}

impl HttpClient {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub async fn post_json(
        &self,
        url: &str,
        headers: Vec<(String, String)>,
        body: Value,
    ) -> Result<Value> {

        let mut req = self.client.post(url);

        for (k, v) in headers {
            req = req.header(k, v);
        }

        let res = req.json(&body).send().await?;

        let json = res.json::<Value>().await?;

        Ok(json)
    }
}