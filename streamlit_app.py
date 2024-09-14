import streamlit as st
import os
from datetime import datetime
import re
from typing import Any, Type
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.callbacks import BaseCallbackHandler
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import json
import wikipedia

original_bs = wikipedia.BeautifulSoup
wikipedia.BeautifulSoup = lambda html: BeautifulSoup(html, features="lxml")

st.set_page_config(
    page_title="Assistants-GPT",
    page_icon="🐼",
)

st.title("🐼 Assistants-GPT")
st.write("왼쪽 설정 창에 OpenAPI API키를 입력해주세요")

# 세션 상태 초기화
session_defaults = {
    "messages": [],
    "api_key": None,
    "api_key_check": False,
}

for key, default in session_defaults.items():
    st.session_state.setdefault(key, default)

API_KEY_PATTERN = r"sk-.*"
API_KEY_ERROR = "API_KEY를 입력하세요"
INVALID_API_KEY = "유효하지 않은 API_KEY입니다"


def wikipedia_search_tool(inputs):
    wk = WikipediaSearchTool()
    query = inputs["query"]
    result = wk.run(query)
    return json.dumps({"result": result}, ensure_ascii=False)


def duckduckgo_search_tool(inputs):
    ddg = DuckDuckGoSearchTool()
    query = inputs["query"]
    result = ddg.run(query)
    return json.dumps({"result": result})


def save_file(docs, title):
    f = open(f"./{title}.txt", "w", encoding="utf-8")
    f.write(docs)


def search_wikipedia(keyword):
    result = "Wikipedia Result\n\n"
    retriver = WikipediaRetriever(top_k_results=3, lang="ko")
    data_list = retriver.invoke(keyword)
    for page_content in data_list:
        result += f"{page_content.page_content} \n\n"
    return result


class WikipediaToolArgsSchema(BaseModel):
    query: str = Field(description="The query you want to search on Wikipedia")


class WikipediaSearchTool(BaseTool):
    name = "WikipediaSearchTool"
    description = """
    Use this tool to search information on Wikipedia. Use 'keyword' as a parameter.
    """
    args_schema: Type[WikipediaToolArgsSchema] = WikipediaToolArgsSchema

    def _run(self, query: str, **kwargs: Any):
        wiki = search_wikipedia(query)
        return wiki


class DuckDuckGoToolArgsCshema(BaseModel):
    query: str = Field(description="The query you want to search on DuckDuckGo")


class DuckDuckGoSearchTool(BaseTool):
    name = "DuckDuckGoSearchTool"
    description = """
    Use this tool to search information on DuckDuckGo. Use 'keyword' as a parameter.
    """
    args_schema: Type[DuckDuckGoToolArgsCshema] = DuckDuckGoToolArgsCshema

    def _scrape_website(self, url: str):
        """Scrapes content from the given URL."""
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            page_text = soup.get_text(separator="\n")
            return page_text
        else:
            return f"Failed to access {url}"

    def _run(self, query: str, **kwargs: Any):
        ddg = DuckDuckGoSearchAPIWrapper()
        result = ddg.results(query, max_results=3)
        print(result)
        if result and len(result) > 0:
            first_link = result[0]["link"]
            scraped_content = self._scrape_website(first_link)
            save_file(scraped_content, "ddg_scraped_content")
            return scraped_content
        else:
            return "No results found or no website link found in DuckDuckGo search."


# 콜백 핸들러 클래스 정의
class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# 메시지 저장 함수
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


# 메시지 전송 함수
def show_message(message, role, save=True, download=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


# 다운로드 버튼 생성 함수
def make_download_button(text, file_path):
    try:
        research_date = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs("./outputs", exist_ok=True)
        file_name = f"./outputs/{file_path}_{research_date}.txt"

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(text)

        return st.download_button(
            label="다운로드",
            data=text,
            file_name=f"{file_path}_{research_date}.txt",
            mime="text/plain",
            key=f"{file_path}_{research_date}",
        )
    except Exception as e:
        st.error(f"An error occurred while creating the download button: {e}")


# 채팅 기록 표시 함수
def paint_history():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for index, message in enumerate(st.session_state["messages"]):
        show_message(
            message["message"],
            message["role"],
            save=False,
            download=False,
        )
        if message["role"] == "assistant":
            make_download_button(
                text=message["message"],
                file_path="result",
            )


# API 키 저장 함수
def save_api_key():
    if re.match(API_KEY_PATTERN, st.session_state["api_key"]):
        st.session_state["api_key_check"] = True
        st.success("API_KEY가 저장되었습니다.")
    else:
        st.error(INVALID_API_KEY)
        st.session_state["api_key_check"] = False


functions_map = {
    "wikipedia_search_tool": wikipedia_search_tool,
    "duckduckgo_search_tool": duckduckgo_search_tool,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search_tool",
            "description": "Use this tool to search information on Wikipedia. Use 'query' as a parameter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you want to search on Wikipedia",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search_tool",
            "description": "Use this tool to search information on DuckDuckGo. Use 'query' as a parameter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you want to search on DuckDuckGo",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


if st.session_state["api_key_check"]:
    client = OpenAI(api_key=st.session_state["api_key"])

    @st.cache_data
    def create_assistant():
        return client.beta.assistants.create(
            name="Research Assistant",
            instructions=(
                "You are a helpful assistant that uses provided tools to answer the user's questions. "
                "When a user asks for information, you should use the appropriate tool to find the information. "
                "Use the 'wikipedia_search_tool' or 'duckduckgo_search_tool' as needed. "
                "If the results contain a URL, extract the content from the link."
            ),
            model="gpt-4o-mini-2024-07-18",
            tools=functions,
        )

    @st.cache_data
    def create_thread(content):
        return client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ]
        )

    @st.cache_data
    def create_run(thread_id, assistant_id):
        return client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )

    def get_run(run_id, thread_id):
        return client.beta.threads.runs.retrieve(
            run_id=run_id,
            thread_id=thread_id,
        )

    def send_message(thread_id, content):
        return client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=content
        )

    def get_messages(thread_id):
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        messages = list(messages)
        messages.reverse()
        assistant_message = None
        for message in messages:
            if message.role == "assistant":
                assistant_message = message.content[0].text.value
        return assistant_message

    def get_tool_outputs(run_id, thread_id):
        run = get_run(run_id, thread_id)
        outputs = []

        for action in run.required_action.submit_tool_outputs.tool_calls:
            action_id = action.id
            function = action.function
            outputs.append(
                {
                    "output": functions_map[function.name](
                        json.loads(function.arguments)
                    ),
                    "tool_call_id": action_id,
                }
            )
        return outputs

    def submit_tool_outputs(run_id, thread_id):
        outpus = get_tool_outputs(run_id, thread_id)
        return client.beta.threads.runs.submit_tool_outputs(
            run_id=run_id, thread_id=thread_id, tool_outputs=outpus
        )

    try:
        assistant = create_assistant()
        if "message" not in st.session_state:
            st.session_state["message"] = ""

        show_message(
            "준비됐습니다. 검색하실 내용을 입력해주세요",
            "assistant",
            save=False,
            download=False,
        )
        paint_history()

        if message := st.chat_input(
            "What do you want reaearch about?", key="message_input"
        ):
            st.session_state["message"] = message

            show_message(message, "user")

            thread = create_thread(message)
            run = create_run(thread.id, assistant.id)

            is_new_result = False

            with st.chat_message("assistant"):
                with st.status(":blue[Polling Run Status...]") as status:

                    while True:
                        run = get_run(run.id, thread.id)
                        if run.status == "requires_action":
                            status.update(
                                label=f"Running: {run.status}", state="running"
                            )
                            submit_tool_outputs(run.id, thread.id)

                        if run.status in ("expired", "cancelled", "failed"):
                            st.write(run.last_error)
                            status.update(
                                label=f":red[{run.status}]",
                                state="error",
                                expanded=True,
                            )
                            break

                        if run.status == "completed":
                            is_new_result = True
                            status.update(label=run.status, state="complete")
                            break

            if is_new_result:
                result = get_messages(thread.id)
                if result:
                    show_message(result, "assistant")
                else:
                    show_message("No result found", "assistant")
                st.rerun()

    except Exception as e:
        st.error(e)


with st.sidebar:
    st.text_input(
        "API_KEY 입력",
        placeholder="OpenAPI API_KEY",
        on_change=save_api_key,
        key="api_key",
        type="password",
    )

    if st.session_state["api_key_check"]:
        st.success("API_KEY가 저장되었습니다.")
    else:
        st.warning(API_KEY_ERROR)

    st.divider()
    st.link_button(
        "Github Repo 바로가기", "https://github.com/asuracoder91/assistants-gpt"
    )


if not st.session_state["api_key_check"]:
    st.warning(API_KEY_ERROR)
    st.stop()
