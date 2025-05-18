**에이전트(Agent)** 모듈
에이전트는 Langchain에서 가장 강력하고 흥미로운 기능 중 하나로, LLM(대규모 언어 모델)을 단순한 텍스트 생성기를 넘어 **스스로 생각하고, 도구를 사용하여 복잡한 작업을 수행하는 자율적인 행위자**로 만들어줍니다.
추론을 하도록 설정

-----

# 1\. 에이전트(Agent) 기본 개념 및 작동 원리 🤖

**에이전트란 무엇인가?**

Langchain에서 에이전트는 LLM을 핵심 \*\*추론 엔진(reasoning engine)\*\*으로 사용합니다. 사용자의 목표가 주어지면, 에이전트는 LLM을 통해 다음 행동(action)을 결정하고, 필요한 경우 \*\*도구(Tools)\*\*를 사용하여 정보를 얻거나 특정 작업을 수행합니다. 그리고 그 도구 사용의 결과(observation)를 다시 LLM에게 전달하여 상황을 판단하고, 목표를 달성할 때까지 이 과정을 반복합니다.

**핵심 구성 요소:**

1.  **LLM / Chat Model**: 에이전트의 "뇌" 역할을 하며, 상황을 판단하고, 어떤 도구를 사용할지, 어떤 입력을 전달할지, 최종 답변은 무엇일지 등을 결정합니다.
2.  **Tools (도구)**: 에이전트가 사용할 수 있는 특정 기능을 가진 함수나 다른 체인입니다.
      * 예시: 웹 검색, 계산기, 데이터베이스 조회, 특정 API 호출, 다른 Langchain 체인 실행 등.
      * 각 도구는 명확한 **이름(name)**, **설명(description)**, 그리고 실행될 \*\*함수(func)\*\*를 가집니다. 설명은 LLM이 이 도구를 언제, 어떻게 사용해야 하는지 이해하는 데 매우 중요합니다.
3.  **Agent Executor (에이전트 실행기)**: 에이전트의 실제 실행 루프를 담당합니다.
      * LLM에게 어떤 도구를 사용할지 묻고 (LLM은 생각과 함께 도구 이름 및 입력 반환),
      * 선택된 도구를 해당 입력으로 실행하고,
      * 도구 실행 결과를 다시 LLM에게 전달하여 다음 단계를 결정하도록 합니다.
      * 이 과정은 최종 답변이 나올 때까지 또는 최대 반복 횟수에 도달할 때까지 반복됩니다.
4.  **Prompt (에이전트 프롬프트)**: LLM이 효과적으로 추론하고 도구를 선택할 수 있도록 특별히 설계된 프롬프트입니다.
      * 여기에는 사용 가능한 도구 목록과 각 도구의 설명, 사용자의 질문/목표, 이전 단계의 생각/행동/관찰 결과 등이 포함됩니다.
      * **ReAct (Reasoning and Acting)** 프레임워크가 자주 사용되며, LLM이 "생각(Thought) -\> 행동(Action) -\> 관찰(Observation)" 사이클을 명시적으로 따르도록 유도합니다.

**일반적인 작동 루프 (ReAct 프레임워크 기반):**

1.  **Thought (생각)**: 에이전트(LLM)는 현재 목표와 주어진 정보를 바탕으로 다음에 무엇을 해야 할지 계획합니다.
2.  **Action (행동)**: 사용할 도구의 이름과 그 도구에 전달할 입력을 결정합니다. 만약 목표를 달성했다고 판단하면, 최종 답변을 생성합니다.
3.  **Observation (관찰)**: 선택된 도구가 실행되고, 그 결과가 관찰 값으로 에이전트에게 돌아옵니다.
4.  (1\~3단계 반복)
5.  **Final Answer (최종 답변)**: 목표가 달성되면, 에이전트는 사용자에게 최종 답변을 제공합니다.

-----

# 2\. 주요 에이전트(Agent) 종류, 특징, 장단점 및 코드 예시

Langchain은 다양한 사전 정의된 에이전트 유형을 제공하며, 필요에 따라 커스텀 에이전트를 만들 수도 있습니다.

## 2.1. `ZeroShotAgent` (또는 `zero-shot-react-description`)

  * **특징**: 가장 일반적이고 기본적인 ReAct 기반 에이전트입니다. LLM이 오직 도구의 \*\*설명(description)\*\*만을 보고 어떤 도구를 사용할지 "제로샷(zero-shot)"으로, 즉 별도의 예시 없이 결정합니다.
  * **장점**:
      * 설정이 비교적 간단합니다.
      * 다양한 도구에 일반적으로 적용하기 쉽습니다.
  * **단점**:
      * LLM이 도구 설명을 정확히 이해하고 적절히 활용하는 능력에 크게 의존합니다.
      * 복잡한 상황이나 도구 간의 미묘한 차이를 구분하기 어려울 수 있습니다.
  * **주요 사용 시나리오**: 일반적인 질의응답, 간단한 작업 자동화.
  * **코드 예시**:

<!-- end list -->

```python
import os
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun # DuckDuckGoSearchRun은 API 키 불필요
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains import LLMMathChain

# API 키 설정 (필요시)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# os.environ["SERPAPI_API_KEY"] = "YOUR_SERPAPI_API_KEY" # SerpAPI 사용 시

# LLM 선택
# llm = OpenAI(temperature=0)
chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # 챗 모델 권장

# 도구 정의
search = DuckDuckGoSearchRun()
llm_math_chain = LLMMathChain.from_llm(llm=chat_llm, verbose=True) # 계산을 위한 체인

tools_zero_shot = [
    Tool(
        name="Search", # 웹 검색 도구
        func=search.run,
        description="현재 사건이나 최신 정보, 특정 주제에 대한 일반적인 질문에 답할 때 유용합니다. 날씨, 뉴스, 특정 인물/장소에 대한 정보 검색 시 사용하세요."
    ),
    Tool(
        name="Calculator", # 계산기 도구
        func=llm_math_chain.run, # LLMMathChain은 문자열 형태의 수학 문제를 받아 풀어줌
        description="수학 계산이 필요할 때 사용합니다. 복잡한 수식이나 단어 형태의 수학 문제도 처리할 수 있습니다."
    )
]

# Zero-shot ReAct 에이전트 초기화
# initialize_agent는 내부적으로 AgentExecutor를 생성하고 반환합니다.
zero_shot_agent = initialize_agent(
    tools_zero_shot,
    chat_llm, # 또는 llm
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True, # 실행 과정을 상세히 보여줌
    handle_parsing_errors=True # LLM의 출력이 잘못된 형식이더라도 에러 대신 메시지 출력
)

print("--- ZeroShotAgent 예시 ---")
try:
    # response_zero_shot = zero_shot_agent.invoke("오늘 서울 날씨는 어떤가요? 그리고 2의 10승은 얼마인가요?")
    response_zero_shot = zero_shot_agent.invoke(
        "최근에 발표된 구글의 가장 큰 AI 모델 이름은 무엇이고, 그 모델의 파라미터 수는 500B (5000억)이라고 가정할 때, "
        "이 파라미터 수를 175B (1750억)으로 나눈 값은 얼마인가요?"
    )
    print(f"\n최종 답변: {response_zero_shot['output']}")
except Exception as e:
    print(f"에이전트 실행 중 오류 발생: {e}")
```

## 2.2. `ConversationalAgent` (또는 `chat-conversational-react-description`)

  * **특징**: 대화형 인터페이스에 적합한 ReAct 기반 에이전트입니다. 이전 대화 내용을 기억하기 위해 **메모리(Memory)** 모듈을 활용합니다.
  * **장점**:
      * 대화의 맥락을 유지하면서 도구를 사용할 수 있어 자연스러운 챗봇 경험을 제공합니다.
  * **단점**:
      * 메모리 관리가 필요하며, 대화가 길어질 경우 메모리 관련 문제(토큰 제한, 비용 증가 등)가 발생할 수 있습니다.
  * **주요 사용 시나리오**: 사용자와 여러 턴에 걸쳐 대화하며 도구를 사용하는 챗봇, 개인 비서.
  * **코드 예시**:

<!-- end list -->

```python
from langchain.memory import ConversationBufferMemory

# 메모리 객체 생성
memory_conversational = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# return_messages=True는 Chat 모델과 함께 사용할 때 권장 (메시지 객체 리스트로 저장)

# Conversational ReAct 에이전트 초기화
# tools_zero_shot과 chat_llm은 위에서 정의한 것 사용
conversational_agent = initialize_agent(
    tools_zero_shot, # 동일한 도구 사용
    chat_llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory_conversational, # 메모리 객체 전달
    handle_parsing_errors=True
)

print("\n--- ConversationalAgent 예시 ---")
try:
    print(conversational_agent.invoke("안녕하세요, 제 이름은 에이전트 마스터입니다.")['output'])
    print(conversational_agent.invoke("제 이름이 뭐라고 했죠?")['output']) # 메모리를 통해 이전 대화 기억
    print(conversational_agent.invoke("오늘 날씨를 알려주고, 123 곱하기 456은 얼마인지 계산해주세요.")['output'])
except Exception as e:
    print(f"에이전트 실행 중 오류 발생: {e}")
```

## 2.3. `OpenAI Functions Agent` / `OpenAI Tools Agent`

  * **특징**: OpenAI의 함수 호출(Function Calling) 또는 도구 사용(Tool Using) 기능을 활용하는 에이전트입니다. LLM이 직접 어떤 함수/도구를 어떤 인자(arguments)로 호출할지 구조화된 JSON 형태로 결정합니다. ReAct 방식보다 더 명시적이고 안정적인 도구 사용이 가능할 수 있습니다. (`openai-tools`가 최신 권장 방식입니다.)
  * **장점**:
      * (특히 OpenAI 모델 사용 시) LLM이 도구 사용을 더 정확하고 안정적으로 결정할 가능성이 높습니다.
      * ReAct 방식보다 토큰 사용량이 적을 수 있습니다 (생각-행동-관찰의 반복이 줄어들 수 있음).
      * 병렬 함수/도구 호출(parallel function/tool calling)을 지원하여 여러 도구를 동시에 실행 요청할 수 있습니다.
  * **단점**:
      * OpenAI 모델에 특화된 기능입니다 (다른 LLM 제공자는 유사한 기능을 자체적으로 지원해야 함).
      * 함수/도구 정의를 OpenAI 명세에 맞게 작성해야 합니다.
  * **주요 사용 시나리오**: OpenAI 모델을 사용하여 신뢰성 높은 도구 연동이 필요할 때.
  * **코드 예시 (`openai-tools`)**:

<!-- end list -->

```python
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # MessagesPlaceholder는 메모리 변수용

# chat_llm과 tools_zero_shot은 위에서 정의한 것 사용 (도구 설명이 중요)

# OpenAI Tools Agent를 위한 프롬프트
# 시스템 메시지, 이전 대화 내역(메모리), 사용자 입력, 그리고 에이전트 스크래치패드(중간 생각/행동)를 위한 플레이스홀더 포함
prompt_openai_tools = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You have access to the following tools: {tool_names}. Only use these tools when necessary. Respond to the human's input as best as you can."),
    MessagesPlaceholder(variable_name="chat_history", optional=True), # 이전 대화 (메모리)
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad") # 에이전트의 중간 단계 (LLM의 생각, 도구 호출/결과)
])

# OpenAI Tools Agent 생성
# create_openai_tools_agent는 LLM, 도구, 프롬프트를 받아 기본적인 에이전트 로직(Runnable)을 만듭니다.
openai_tools_agent_runnable = create_openai_tools_agent(chat_llm, tools_zero_shot, prompt_openai_tools)

# 메모리 (선택 사항, 대화형으로 만들고 싶을 때)
memory_openai_tools = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# AgentExecutor 생성
# AgentExecutor는 이 Runnable 에이전트와 도구들을 받아 실제 실행을 담당합니다.
openai_tools_agent_executor = AgentExecutor(
    agent=openai_tools_agent_runnable,
    tools=tools_zero_shot,
    memory=memory_openai_tools, # 메모리 추가
    verbose=True,
    handle_parsing_errors=True
)

print("\n--- OpenAI Tools Agent 예시 ---")
try:
    print(openai_tools_agent_executor.invoke({"input": "테슬라의 현재 주가는 얼마인가요? 그리고 원주율의 첫 5자리는 무엇인가요? (계산기 말고 검색 사용)"})['output'])
    # 검색 도구가 주가 정보를, 다른 검색으로 원주율 정보를 가져올 수 있음
    # (만약 수학 도구를 원주율 계산에 사용하려 한다면 프롬프트나 도구 설명을 조정해야 할 수 있음)
    print(openai_tools_agent_executor.invoke({"input": "물의 끓는점은 섭씨로 얼마인가요? 그리고 10 더하기 20은?"})['output'])

except Exception as e:
    print(f"에이전트 실행 중 오류 발생: {e}")
```

*참고: `OpenAI Functions Agent` (`AgentType.OPENAI_FUNCTIONS` 또는 `create_openai_functions_agent`)는 유사하지만, `tools` 대신 `functions`라는 용어를 사용했던 레거시 방식입니다. 현재는 `tools` 사용이 권장됩니다.*

## 2.4. `SelfAskAgent` (또는 `self-ask-with-search`)

  * **특징**: 복잡한 질문에 답하기 위해, 원본 질문을 여러 개의 간단한 하위 질문으로 나누고, 각 하위 질문에 대해 **검색 도구**를 사용하여 답을 찾습니다. 이렇게 얻은 중간 답변(intermediate answer)들을 종합하여 최종 답변을 도출합니다.
  * **장점**:
      * 다단계 추론이나 여러 정보를 조합해야 하는 사실 기반 질문에 효과적입니다.
      * 검색 결과를 단계적으로 활용하여 문제 해결 과정을 명확히 보여줍니다.
  * **단점**:
      * 검색 도구에 대한 의존도가 매우 높습니다.
      * 질문을 적절히 분해하고 중간 답변을 잘 종합하는 능력이 LLM의 성능에 크게 좌우됩니다.
      * 주로 단일 검색 도구와 함께 사용됩니다.
  * **주요 사용 시나리오**: "X의 아내의 직업은 무엇인가?"와 같이 연쇄적인 정보 검색이 필요한 질문.
  * **코드 예시**:

<!-- end list -->

```python
# 검색 도구 (하나만 사용)
search_tool_for_self_ask = Tool(
    name="Intermediate Answer", # SelfAskAgent는 이 특정 이름의 도구를 찾음
    func=search.run, # DuckDuckGoSearchRun 객체의 run 메서드
    description="검색 엔진. 사실에 기반한 질문에 답할 때 사용."
)

# Self-Ask with Search 에이전트 초기화
self_ask_agent = initialize_agent(
    [search_tool_for_self_ask], # 단일 검색 도구 전달
    chat_llm, # 또는 llm
    agent=AgentType.SELF_ASK_WITH_SEARCH,
    verbose=True,
    handle_parsing_errors=True
)

print("\n--- SelfAskAgent 예시 ---")
try:
    # 이 질문은 여러 단계의 검색을 유도할 수 있음
    response_self_ask = self_ask_agent.invoke("현재 프랑스 대통령의 아내가 태어난 도시는 어디인가요?")
    print(f"\n최종 답변: {response_self_ask['output']}")
except Exception as e:
    print(f"에이전트 실행 중 오류 발생: {e}")
```

## 기타 에이전트 유형 (간략 소개)

  * **XML Agent**: Anthropic Claude 모델과 같이 XML 태그를 사용하여 생각과 도구 사용을 표현하는 데 적합한 모델을 위한 에이전트입니다.
  * **Plan-and-Execute Agent**: 먼저 전체 작업에 대한 계획을 여러 단계로 수립하고(Plan), 그 계획에 따라 각 단계를 순차적으로 실행(Execute)하는 에이전트입니다. 복잡하고 장기적인 목표를 가진 작업에 유용할 수 있습니다.

-----

# 3\. Tools (도구) 정의 방법

에이전트의 성능은 어떤 도구를 얼마나 잘 정의하느냐에 크게 좌우됩니다.

  * **`Tool` 클래스 사용**:

      * `name` (str): 도구의 고유한 이름. 에이전트가 이 이름을 사용하여 도구를 참조합니다.
      * `func` (Callable): 도구의 실제 기능을 수행하는 함수 또는 코루틴. 단일 문자열 입력을 받고 단일 문자열을 반환하는 것이 일반적입니다.
      * `description` (str): **매우 중요\!** LLM이 이 도구를 언제, 어떻게, 왜 사용해야 하는지 명확하고 상세하게 이해할 수 있도록 작성해야 합니다. 좋은 설명은 에이전트의 도구 선택 능력을 크게 향상시킵니다.
      * `return_direct` (bool, 기본값 False): True로 설정하면, 이 도구가 실행된 후 그 결과를 바로 최종 답변으로 반환하고 에이전트 실행을 종료합니다.
      * `args_schema` (Pydantic BaseModel, 선택 사항): 도구의 입력 인자 스키마를 정의하여 LLM이 더 정확한 입력을 생성하도록 유도할 수 있습니다 (특히 OpenAI Functions/Tools Agent와 함께 사용 시 유용).

  * **`@tool` 데코레이터 사용**: 함수를 정의하고 `@tool` 데코레이터를 붙이면 자동으로 `Tool` 객체로 변환됩니다. 함수의 docstring이 `description`으로 사용됩니다.

**코드 예시 (커스텀 도구 정의):**

```python
from langchain_core.tools import tool # 데코레이터 임포트

@tool
def get_word_length(word: str) -> int:
    """입력된 단어의 글자 수를 반환합니다."""
    return len(word)

@tool
def get_current_time_in_seoul() -> str:
    """현재 대한민국의 서울 시간을 '년-월-일 시:분:초' 형식으로 반환합니다."""
    # (실제 구현에서는 datetime 라이브러리와 pytz 등을 사용해야 함)
    # 여기서는 예시로 현재 KST 시간을 하드코딩 (실제로는 동적으로 생성)
    # from datetime import datetime
    # import pytz
    # seoul_tz = pytz.timezone('Asia/Seoul')
    # current_seoul_time = datetime.now(seoul_tz)
    # return current_seoul_time.strftime("%Y-%m-%d %H:%M:%S")
    return "2025-05-19 08:32:35" # 현재 시뮬레이션된 시간

# 위에서 정의한 get_word_length 도구 객체 확인
print(f"\n--- @tool 데코레이터로 생성된 도구 ---")
print(f"도구 이름: {get_word_length.name}")
print(f"도구 설명: {get_word_length.description}")
print(f"도구 기능 실행 (예시): {get_word_length.run('Langchain')}")

# 이 커스텀 도구들을 다른 도구들과 함께 에이전트에 전달할 수 있습니다.
custom_tools = [get_word_length, get_current_time_in_seoul()]
# 예를 들어, tools_zero_shot 리스트에 custom_tools를 추가하여 에이전트에 사용 가능
# all_tools = tools_zero_shot + custom_tools
# ... agent 초기화 시 all_tools 사용 ...
```

-----

# 4\. 에이전트(Agent) 사용 시 고려사항 및 팁

  * **명확하고 상세한 도구 설명**: 에이전트가 도구를 올바르게 이해하고 사용하도록 하는 데 가장 중요한 요소입니다. "이 도구는 X를 할 때 유용하며, Y와 같은 입력을 받아 Z와 같은 출력을 반환합니다."와 같이 구체적으로 작성하세요.
  * **LLM의 추론 능력**: 에이전트의 전반적인 성능은 기반 LLM의 지능과 추론 능력에 크게 좌우됩니다. GPT-4와 같은 고성능 모델이 더 나은 결과를 보이는 경향이 있습니다.
  * **프롬프트 엔지니어링**: 에이전트 프롬프트(prefix, suffix, format\_instructions 등)를 필요에 따라 조정하여 성능을 개선할 수 있습니다. (커스텀 에이전트 작성 시)
  * **오류 처리 및 견고성**:
      * `handle_parsing_errors=True` (또는 커스텀 핸들러)를 사용하여 LLM의 출력이 예상 형식을 벗어났을 때의 오류를 관리합니다.
      * 도구 실행 중 발생할 수 있는 예외를 적절히 처리하도록 도구 함수를 작성합니다.
      * `max_iterations`를 설정하여 에이전트가 무한 루프에 빠지는 것을 방지합니다.
  * **비용 및 속도**: 에이전트는 여러 번의 LLM 호출과 도구 실행을 포함할 수 있으므로, 비용과 응답 시간을 고려해야 합니다.
  * **보안 및 안전성**:
      * 에이전트가 외부 도구나 API를 사용할 때 보안 위험을 인지해야 합니다.
      * 에이전트가 예상치 못한 행동을 하거나 악의적인 입력을 통해 원치 않는 작업을 수행하지 않도록 주의해야 합니다. (Langchain Guardrails와 같은 도구 고려)
      * 특히 사용자가 직접 도구 입력 내용을 제어할 수 있는 경우 위험이 커질 수 있습니다.

-----
