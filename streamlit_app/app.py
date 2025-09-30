"""Gemini-inspired conversational UI with Azure OpenAI backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List

import streamlit as st
from openai import AzureOpenAI


@dataclass
class AzureChatConfig:
    """Configuration required for interacting with Azure OpenAI chat deployments."""

    endpoint: str
    api_key: str
    api_version: str
    deployment: str

    @classmethod
    def from_environment(cls) -> "AzureChatConfig | None":
        """Create configuration from environment variables if available."""

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not all([endpoint, api_key, deployment]):
            return None

        return cls(endpoint=endpoint, api_key=api_key, api_version=api_version, deployment=deployment)


@st.cache_resource(show_spinner=False)
def create_client(config: AzureChatConfig) -> AzureOpenAI:
    """Create and cache an Azure OpenAI client."""

    return AzureOpenAI(
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
        api_version=config.api_version,
    )


def format_history(history: Iterable[dict[str, str]]) -> List[dict[str, str]]:
    """Convert Streamlit session history into Azure OpenAI message objects."""

    system_prompt = {
        "role": "system",
        "content": (
            "You are a friendly and detail oriented assistant chatting inside a Gemini-inspired "
            "interface. Provide concise, helpful answers with markdown support."
        ),
    }

    formatted = [system_prompt]
    for item in history:
        formatted.append({"role": item["role"], "content": item["content"]})
    return formatted


def stream_chat_completion(client: AzureOpenAI, config: AzureChatConfig, history: List[dict[str, str]]):
    """Stream chat completion tokens and yield cumulative content."""

    response = client.chat.completions.create(
        model=config.deployment,
        messages=format_history(history),
        temperature=0.7,
        stream=True,
    )

    partial = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            partial += delta
            yield partial


def init_session_state() -> None:
    """Initialize Streamlit session state for chat messages."""

    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_header() -> None:
    """Render top header to mimic Gemini's floating search bar."""

    st.markdown(
        """
        <div class="gemini-header">
            <div class="gemini-logo">Gemini inspired</div>
            <div class="gemini-pill">Chat with Azure OpenAI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_suggestion_chips() -> None:
    """Render suggestion chips similar to Gemini's quick prompts."""

    suggestions = [
        "最新のAIトレンドを教えて",
        "Azure OpenAIの使い方を解説して",
        "PythonでのStreamlit活用例は？",
        "Azure App Serviceへのデプロイ手順",
    ]

    st.markdown("<div class='suggestion-row'>" + "".join(
        f"<div class='suggestion-chip'>{s}</div>" for s in suggestions
    ) + "</div>", unsafe_allow_html=True)


def render_sidebar(config: AzureChatConfig | None) -> None:
    """Display configuration helper in the sidebar."""

    with st.sidebar:
        st.header("Connection settings")
        st.write(
            "環境変数でAzure OpenAIの接続情報を設定してください。"
            "App Serviceでは[構成]から設定するのが便利です。"
        )
        st.code(
            """
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="<your-key>"
export AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o-mini"
export AZURE_OPENAI_API_VERSION="2024-02-01"
            """.strip()
        )

        st.divider()
        st.markdown("### 現在のステータス")
        if config is None:
            st.error("環境変数が不足しています。キーとエンドポイントを設定してください。")
        else:
            st.success("Azure OpenAIに接続可能な状態です。")
            st.json(
                {
                    "endpoint": config.endpoint,
                    "deployment": config.deployment,
                    "api_version": config.api_version,
                }
            )

        st.markdown("""
---
- 新しいチャットを始めるには「New chat」を押してください。
- メッセージはセッション内にのみ保存されます。
        """)


def render_chat(config: AzureChatConfig | None) -> None:
    """Render chat body and handle user interactions."""

    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="✨"):
            st.markdown("こんにちは！Gemini風の体験でAzure OpenAIと会話できます。何をお手伝いしましょうか？")

    for message in st.session_state.messages:
        avatar = "✨" if message["role"] == "assistant" else "🧑"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("メッセージを入力してください..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        if config is None:
            with st.chat_message("assistant", avatar="✨"):
                st.error("Azure OpenAIの設定が見つかりません。環境変数を設定してください。")
            return

        client = create_client(config)
        with st.chat_message("assistant", avatar="✨"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                for partial in stream_chat_completion(client, config, st.session_state.messages):
                    full_response = partial
                    message_placeholder.markdown(full_response + "▌")
            except Exception as exc:  # pragma: no cover - Streamlit UI feedback
                message_placeholder.error(f"エラーが発生しました: {exc}")
                return

            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def main() -> None:
    st.set_page_config(page_title="Gemini inspired chat", page_icon="✨", layout="wide")

    with open(os.path.join(os.path.dirname(__file__), "styles.css"), encoding="utf-8") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

    init_session_state()

    config = AzureChatConfig.from_environment()
    render_sidebar(config)

    render_header()
    st.markdown("""
        <div class="gemini-hero">
            <div>
                <h1>どんなことでもお任せください</h1>
                <p>Geminiの雰囲気でAzure OpenAIと会話できます。質問やアイデアを入力して、リアルタイムで回答を受け取りましょう。</p>
            </div>
            <div class="hero-badge">Azure OpenAI</div>
        </div>
    """, unsafe_allow_html=True)

    render_suggestion_chips()

    st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)
    new_chat, spacer = st.columns([0.2, 0.8])
    with new_chat:
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.experimental_rerun()
    spacer.empty()

    render_chat(config)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
