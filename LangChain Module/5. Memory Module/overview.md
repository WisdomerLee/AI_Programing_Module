**메모리(Memory) 모듈**
메모리는 대화형 AI 애플리케이션에서 이전 대화의 맥락을 "기억"하여 보다 자연스럽고 일관된 상호작용을 가능하게 하는 핵심 요소입니다.

-----

# 1\. 메모리(Memory)의 역할 및 중요성

챗봇이나 대화형 에이전트는 사용자와 여러 턴(turn)에 걸쳐 대화를 나눕니다. 이때 이전 대화 내용을 기억하지 못한다면 매번 새로운 대화를 시작하는 것처럼 부자연스러울 것입니다. Langchain의 메모리 모듈은 다음과 같은 역할을 수행합니다:

  * **맥락 유지**: 이전 대화의 내용을 저장하여 현재 대화에 활용합니다.
  * **일관성 있는 답변**: 사용자가 이전에 언급했던 정보(이름, 선호도 등)를 기억하고 답변에 반영합니다.
  * **상태 저장(Stateful) 애플리케이션**: 대화의 상태를 관리하여 보다 복잡한 상호작용을 가능하게 합니다.

-----

# 2\. 메모리의 기본 작동 방식

Langchain의 체인(Chain)이나 에이전트(Agent)는 메모리 객체와 함께 작동합니다.

1.  **실행 전 (Load)**: 체인/에이전트가 실행되기 전에 메모리에서 과거 대화 기록을 불러옵니다. 이 기록은 프롬프트의 일부로 LLM에 전달되어 LLM이 맥락을 이해하도록 돕습니다.
2.  **실행 후 (Save)**: 체인/에이전트가 실행된 후, 현재 대화의 입력과 출력이 메모리에 저장됩니다.

<!-- end list -->

  * **`memory_key`**: 프롬프트 템플릿 내에서 메모리 변수(예: 이전 대화 기록)를 참조할 때 사용하는 키입니다. (예: `history`, `chat_history`)
  * **`input_key` / `output_key`**: (일부 메모리 유형에서 사용) 어떤 입력과 출력을 메모리에 기록할지 명시적으로 지정할 때 사용됩니다.

-----

# 3\. 주요 메모리 종류, 특징, 장단점 및 코드 예시

Langchain은 다양한 종류의 메모리를 제공하여 상황에 맞게 선택할 수 있도록 합니다.

## 3.1. `ConversationBufferMemory`

  * **특징**: 대화의 전체 내용을 있는 그대로 순서대로 버퍼에 저장합니다. 가장 간단하고 직관적인 메모리입니다.
  * **장점**:
      * 구현이 매우 쉽습니다.
      * 모든 대화 내용을 정확하게 보존합니다.
      * 짧은 대화나 모든 세부 정보가 중요한 경우에 적합합니다.
  * **단점**:
      * 대화가 길어지면 버퍼의 크기가 매우 커져 LLM의 컨텍스트 윈도우(토큰 제한)를 초과할 수 있습니다.
      * 토큰 제한 초과 시 오류가 발생하거나, 비용이 크게 증가할 수 있습니다.
  * **주요 파라미터**:
      * `memory_key` (str): 프롬프트에서 사용할 대화 기록 변수 이름 (기본값: "history").
      * `return_messages` (bool): True이면 메시지 객체 리스트로, False이면 문자열로 기록을 반환 (기본값: False).

**코드 예시:**

```python
import os
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# API 키 설정 (실제 키로 대체하거나 환경 변수로 설정하세요)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# ConversationBufferMemory 사용
# return_messages=True로 설정하면 Chat Model과 함께 사용하기 용이
memory_buffer = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

# 프롬프트 템플릿에 memory_key (chat_history) 포함
# ConversationChain은 내부적으로 적절한 프롬프트를 사용하지만, 커스텀 프롬프트도 가능
# 여기서는 ConversationChain의 기본 프롬프트를 사용하도록 설정
conversation_buffer_chain = ConversationChain(
    llm=llm,
    memory=memory_buffer,
    verbose=True # 실행 과정 로깅
)

print("--- ConversationBufferMemory 예시 ---")
print(conversation_buffer_chain.invoke("안녕하세요, 제 이름은 알파입니다.")['response'])
# 메모리 내용: Human: 안녕하세요, 제 이름은 알파입니다. AI: 안녕하세요 알파님! 만나서 반갑습니다. 무엇을 도와드릴까요?

print(conversation_buffer_chain.invoke("오늘 날씨가 어떤가요?")['response'])
# 메모리 내용: (이전 내용) Human: 오늘 날씨가 어떤가요? AI: 죄송하지만, 저는 실시간 날씨 정보를 제공할 수 없습니다...

print(conversation_buffer_chain.invoke("제 이름을 기억하시나요?")['response'])
# 메모리 내용: (이전 내용) Human: 제 이름을 기억하시나요? AI: 네, 알파님이라고 기억합니다.

print("\n최종 메모리 내용 (Buffer):")
print(memory_buffer.buffer) # 또는 memory_buffer.load_memory_variables({})['chat_history']
```

## 3.2. `ConversationBufferWindowMemory`

  * **특징**: 최근 `k`개의 대화 턴(turn)만 버퍼에 저장합니다. 오래된 대화는 순차적으로 삭제됩니다.
  * **장점**:
      * `ConversationBufferMemory`의 토큰 제한 문제를 완화합니다.
      * 최근 대화의 맥락에 집중할 수 있습니다.
  * **단점**:
      * `k`개 이전의 오래된 대화 내용은 손실됩니다.
      * 중요한 정보가 오래된 대화에 있었다면 놓칠 수 있습니다.
  * **주요 파라미터**:
      * `k` (int): 기억할 최근 대화 턴의 수.
      * `memory_key`, `return_messages` 등 `ConversationBufferMemory`와 유사.

**코드 예시:**

```python
from langchain.memory import ConversationBufferWindowMemory

# k=2로 설정하여 최근 2개의 대화 턴만 기억
memory_window = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=False)

conversation_window_chain = ConversationChain(
    llm=llm,
    memory=memory_window,
    verbose=True
)

print("\n--- ConversationBufferWindowMemory (k=2) 예시 ---")
print(conversation_window_chain.invoke("안녕하세요, 제 이름은 베타입니다.")['response']) # 1번째 턴
print(conversation_window_chain.invoke("저는 파란색을 좋아합니다.")['response'])          # 2번째 턴
print(conversation_window_chain.invoke("오늘의 추천 메뉴는 무엇인가요?")['response'])      # 3번째 턴 (1번째 턴 내용 일부 손실 시작 가능성 - 실제로는 AI 응답까지 한 턴)

# 4번째 턴: 베타라는 이름은 기억하지만, 파란색 정보는 밀려날 수 있음
print(conversation_window_chain.invoke("제 이름과 좋아하는 색을 기억하시나요?")['response'])

print("\n최종 메모리 내용 (Window, k=2):")
print(memory_window.buffer)
```

*참고: `k`는 Human-AI 상호작용 쌍(turn)을 의미합니다. `return_messages=False`일 경우 Human/AI 메시지가 번갈아 문자열로 저장됩니다. `k=2`이면 최근 4개 메시지(2쌍)가 저장될 수 있습니다.*

## 3.3. `ConversationTokenBufferMemory`

  * **특징**: 최근 대화 내용 중 총 토큰 수가 특정 제한(`max_token_limit`)을 넘지 않도록 저장합니다. 토큰 제한을 초과하면 가장 오래된 메시지부터 삭제합니다.
  * **장점**:
      * LLM의 토큰 제한에 맞춰 메모리 크기를 정교하게 제어할 수 있습니다.
      * 비용 관리에 유리합니다.
  * **단점**:
      * 토큰 수를 계산하기 위해 LLM 객체(`llm`)가 필요합니다. (내부적으로 `tiktoken` 등 사용)
  * **주요 파라미터**:
      * `llm`: 토큰 수를 계산하는 데 사용될 LLM 객체.
      * `max_token_limit` (int): 메모리에 저장할 최대 토큰 수.

**코드 예시:**

```python
from langchain.memory import ConversationTokenBufferMemory

# 예시 LLM (OpenAI 모델은 tiktoken 라이브러리를 통해 토큰 수 계산 가능)
# max_token_limit을 작게 설정하여 효과를 쉽게 확인
memory_token = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=50, # 매우 작은 토큰 제한 (실제로는 더 크게 설정)
    memory_key="chat_history",
    return_messages=False
)

conversation_token_chain = ConversationChain(
    llm=llm,
    memory=memory_token,
    verbose=True
)

print("\n--- ConversationTokenBufferMemory (max_token_limit=50) 예시 ---")
print(conversation_token_chain.invoke("안녕! 내 이름은 감마야. 나는 여행을 아주 좋아해.")['response'])
# 위 문장은 토큰 수가 꽤 될 것입니다.

print(conversation_token_chain.invoke("최근에 다녀온 여행지는 어디야?")['response'])
# 첫 번째 대화 내용이 토큰 제한으로 인해 일부 또는 전체가 잘릴 수 있습니다.

print(conversation_token_chain.invoke("내 이름을 기억하니?")['response'])

print("\n최종 메모리 내용 (Token Buffer):")
# memory_token.buffer는 메시지 객체 리스트일 수 있음 (return_messages 값에 따라)
# 문자열로 보려면 load_memory_variables 사용
print(memory_token.load_memory_variables({})['chat_history'])
```

## 3.4. `ConversationSummaryMemory`

  * **특징**: 대화가 진행됨에 따라, 이전 대화 내용을 LLM을 사용하여 요약하고, 이 요약본을 메모리에 저장합니다. 새로운 대화가 추가되면 전체 대화에 대한 새로운 요약을 생성합니다.
  * **장점**:
      * 대화가 매우 길어져도 요약된 형태로 핵심 정보를 보존하므로 토큰 효율성이 매우 높습니다.
      * 장기적인 대화의 맥락 유지에 유리합니다.
  * **단점**:
      * 매번 요약을 위해 LLM 호출이 추가로 발생하여 비용과 응답 시간이 증가할 수 있습니다.
      * 요약 과정에서 세부 정보가 손실될 가능성이 있습니다. LLM의 요약 성능에 의존합니다.
  * **주요 파라미터**:
      * `llm`: 대화 내용을 요약하는 데 사용될 LLM 객체.

**코드 예시:**

```python
from langchain.memory import ConversationSummaryMemory

# 요약용 LLM
summary_llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
memory_summary = ConversationSummaryMemory(llm=summary_llm, memory_key="chat_summary")

# 프롬프트에 요약된 내용이 들어갈 부분을 명시해야 함
# ConversationChain은 memory_key를 기반으로 이를 처리
prompt_summary = PromptTemplate(
    input_variables=["chat_summary", "input"], # 'chat_summary'가 메모리 키
    template="""다음은 사용자와 AI 간의 친근한 대화 요약입니다.
{chat_summary}

현재 대화:
사용자: {input}
AI:"""
)

conversation_summary_chain = ConversationChain(
    llm=llm, # 답변 생성용 LLM
    memory=memory_summary,
    prompt=prompt_summary, # 커스텀 프롬프트 사용
    verbose=True
)

print("\n--- ConversationSummaryMemory 예시 ---")
print(conversation_summary_chain.invoke("안녕하세요, 저는 델타입니다. 고양이를 키우고 있어요.")['response'])
# 메모리: "사용자는 델타라고 자신을 소개했고 고양이를 키운다고 말했습니다." (LLM이 생성한 요약)

print(conversation_summary_chain.invoke("제 고양이 이름은 나비예요. 나비는 매우 활발해요.")['response'])
# 메모리: "사용자 델타는 나비라는 이름의 활발한 고양이를 키우고 있습니다." (이전 요약 + 새 정보로 업데이트된 요약)

print(conversation_summary_chain.invoke("제가 키우는 애완동물이 뭐라고 했죠?")['response'])
# AI는 요약된 내용을 바탕으로 답변

print("\n최종 메모리 내용 (Summary):")
print(memory_summary.buffer) # 요약된 내용
```

## 3.5. `ConversationSummaryBufferMemory`

  * **특징**: `ConversationSummaryMemory`와 `ConversationTokenBufferMemory` (또는 `ConversationBufferWindowMemory`)의 장점을 결합한 하이브리드 방식입니다. 최근 대화는 그대로 버퍼에 저장하고, 오래되어 버퍼에서 밀려나는 대화는 요약하여 별도의 요약 메모리에 보관합니다.
  * **장점**:
      * 최근 대화의 세부 정보와 오래된 대화의 장기적인 맥락을 모두 유지할 수 있습니다.
      * 토큰 효율성과 정보 보존 간의 균형을 잘 맞출 수 있습니다.
  * **단점**:
      * 설정이 상대적으로 복잡하며, 요약용 LLM과 토큰 계산을 위한 설정이 모두 필요합니다.
  * **주요 파라미터**:
      * `llm`: 요약용 LLM.
      * `max_token_limit`: 버퍼에 저장할 최근 대화의 최대 토큰 수.
      * `memory_key`, `human_prefix`, `ai_prefix` 등.

**코드 예시:**

```python
from langchain.memory import ConversationSummaryBufferMemory

# 요약용 LLM, 답변 생성용 LLM은 위에서 정의한 llm, summary_llm 사용
memory_summary_buffer = ConversationSummaryBufferMemory(
    llm=summary_llm, # 요약용 LLM
    max_token_limit=60, # 최근 대화 버퍼의 토큰 제한 (작게 설정하여 효과 확인)
    memory_key="history",
    return_messages=False # 또는 True
)

conversation_summary_buffer_chain = ConversationChain(
    llm=llm, # 답변 생성용 LLM
    memory=memory_summary_buffer,
    verbose=True
)

print("\n--- ConversationSummaryBufferMemory (max_token_limit=60) 예시 ---")
print(conversation_summary_buffer_chain.invoke("제 이름은 엡실론이고, 취미는 독서입니다.")['response'])
print(conversation_summary_buffer_chain.invoke("가장 최근에 읽은 책은 '멋진 신세계'입니다. 아주 인상 깊었어요.")['response'])
# 위 대화들은 max_token_limit을 초과할 가능성이 높음. 일부는 버퍼에, 오래된 부분은 요약으로 넘어감.

print(conversation_summary_buffer_chain.invoke("제 이름과 취미, 그리고 최근 읽은 책을 기억하시나요?")['response'])

print("\n최종 메모리 내용 (Summary Buffer):")
# memory_summary_buffer.load_memory_variables({})로 확인 가능
# 내부적으로 moving_summary_buffer (요약)와 chat_memory (최근 버퍼)를 관리
print(f"요약된 부분: {memory_summary_buffer.moving_summary_buffer}")
print(f"최근 버퍼 부분: {memory_summary_buffer.chat_memory.messages}")
```

## 3.6. `VectorStoreRetrieverMemory` (고급)

  * **특징**: 대화 내용을 벡터 스토어(Vector Store)에 임베딩하여 저장합니다. 필요시 현재 대화와 관련된 과거 대화 조각을 벡터 유사도 검색을 통해 가져와 컨텍스트로 활용합니다.
  * **장점**:
      * 매우 긴 대화나 방대한 양의 과거 정보를 효율적으로 검색하고 활용할 수 있습니다.
      * 의미 기반 검색을 통해 단순히 시간 순서가 아닌, 현재 대화와 관련성 높은 정보를 정확하게 찾아낼 수 있습니다.
  * **단점**:
      * 설정이 복잡합니다 (임베딩 모델, 벡터 스토어 구성 필요).
      * 검색 품질이 전체 성능에 큰 영향을 미칩니다.
      * 매번 검색 및 임베딩 과정에서 약간의 지연이 발생할 수 있습니다.
  * **주요 구성 요소**:
      * `retriever`: 벡터 스토어 기반의 리트리버 객체.

**코드 예시 (개념 및 간단한 인메모리 구현):**

```python
from langchain_community.vectorstores import FAISS # langchain-community, faiss-cpu/gpu 설치 필요
from langchain_openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory

# VectorStoreRetrieverMemory는 대화 턴을 문서로 저장하고 검색합니다.
# 실제로는 각 대화 턴을 Document 객체로 만들어 저장합니다.

# 1. 임베딩 모델 및 벡터 스토어 준비 (간단한 인메모리 FAISS)
try:
    embeddings = OpenAIEmbeddings()
    # FAISS는 초기화 시 문서가 필요할 수 있으므로, 빈 Document로 시작하거나 나중에 추가
    # 여기서는 retriever를 먼저 만들고, 메모리 객체가 내부적으로 문서를 추가/검색하도록 함
    # VectorStoreRetrieverMemory는 내부적으로 VectorStore를 생성하거나 기존 것을 사용합니다.

    # VectorStoreRetrieverMemory는 직접 vectorstore를 만들거나 retriever를 받습니다.
    # 여기서는 간단히 FAISS를 직접 만들고 retriever를 전달합니다.
    # 실제 사용 시에는 더 견고한 VectorStore 설정이 필요합니다.
    index = FAISS.from_texts(["초기화용 더미 텍스트"], embeddings) # FAISS 초기화
    retriever = index.as_retriever(search_kwargs=dict(k=1)) # 가장 관련성 높은 1개 검색

    # 2. VectorStoreRetrieverMemory 생성
    vstore_memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="relevant_history")

    # 3. 메모리에 대화 내용 저장 (save_context)
    # ConversationChain 등에서 자동으로 호출되지만, 여기서는 수동으로 시연
    vstore_memory.save_context(
        {"input": "저는 제타이고, 축구를 좋아합니다."},
        {"output": "안녕하세요 제타님! 축구를 좋아하시는군요."}
    )
    vstore_memory.save_context(
        {"input": "가장 좋아하는 축구팀은 맨체스터 유나이티드입니다."},
        {"output": "맨체스터 유나이티드를 좋아하시는군요! 멋진 팀이죠."}
    )
    vstore_memory.save_context(
        {"input": "저는 프로그래밍도 공부하고 있어요."},
        {"output": "프로그래밍 공부도 하시는군요! 어떤 언어를 주로 사용하시나요?"}
    )


    # 4. 현재 입력에 대해 관련 과거 대화 로드 (load_memory_variables)
    relevant_history = vstore_memory.load_memory_variables({"input": "제가 좋아하는 스포츠가 뭐라고 했죠?"})['relevant_history']

    print("\n--- VectorStoreRetrieverMemory 예시 ---")
    print(f"현재 질문: 제가 좋아하는 스포츠가 뭐라고 했죠?")
    print(f"검색된 관련 과거 대화: {relevant_history}")
    # 출력 예상: "Human: 저는 제타이고, 축구를 좋아합니다.\nAI: 안녕하세요 제타님! 축구를 좋아하시는군요."

    # 실제 체인과 연동 예시 (간단화)
    prompt_vstore = PromptTemplate(
        input_variables=["relevant_history", "input"],
        template="""다음은 대화의 관련 부분입니다:
    {relevant_history}

    사용자: {input}
    AI:"""
    )
    chain_vstore = ConversationChain(llm=llm, memory=vstore_memory, prompt=prompt_vstore, verbose=True)
    print(chain_vstore.invoke("제가 좋아하는 스포츠가 뭐라고 했는지 다시 알려주세요.")['response'])


except ImportError:
    print("\nFAISS 또는 관련 라이브러리가 설치되지 않았습니다. `pip install faiss-cpu langchain-openai` 등을 실행해주세요.")
except Exception as e:
    print(f"\nVectorStoreRetrieverMemory 예외 발생: {e}. API 키 등을 확인해주세요.")
```

## 3.7. `ConversationKGMemory` (지식 그래프 메모리 - 고급)

  * **특징**: 대화에서 엔티티(개체)와 그 관계를 추출하여 지식 그래프(Knowledge Graph) 형태로 저장합니다. 이를 통해 구조화된 방식으로 대화의 맥락을 이해하고 활용합니다.
  * **장점**:
      * 대화 속 엔티티 간의 복잡한 관계를 명시적으로 파악하고 추론에 활용할 수 있습니다.
      * 단순 키워드 매칭을 넘어선 깊이 있는 맥락 이해가 가능할 수 있습니다.
  * **단점**:
      * 지식 그래프 구축 및 관리가 매우 복잡하며, 정확한 정보 추출을 위해 강력한 LLM 성능이 요구됩니다.
      * 구현 난이도가 높고, 아직 활발히 연구 개발 중인 분야입니다.
  * **주요 파라미터**:
      * `llm`: 지식 그래프 트리플(주어-관계-목적어)을 추출하는 데 사용될 LLM.

**코드 예시 (개념 위주, 실제 구현은 더 복잡):**

```python
from langchain.memory import ConversationKGMemory

# 지식 추출용 LLM (예시, 실제로는 이 작업에 특화된 모델이나 프롬프트 필요)
kg_llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
memory_kg = ConversationKGMemory(llm=kg_llm, memory_key="kg_history") # 기본 프롬프트 사용

# ConversationChain과 연동
# KG 메모리는 추출된 지식 그래프 정보를 프롬프트에 주입합니다.
# kg_prompt_template = """다음은 지금까지 대화에서 추출한 지식 그래프 요약입니다:
# {kg_history}

# 관련 엔티티: {entities}  <-- ConversationKGMemory가 자동으로 채워주는 부분 (선택적)

# 현재 대화:
# 사용자: {input}
# AI:"""
# kg_prompt = PromptTemplate(input_variables=["kg_history", "entities", "input"], template=kg_prompt_template)

# KGMemory는 내부적으로 get_prompt() 메서드를 통해 적절한 프롬프트를 생성하기도 합니다.
conversation_kg_chain = ConversationChain(
    llm=llm, # 답변 생성용 LLM
    memory=memory_kg,
    verbose=True
)

print("\n--- ConversationKGMemory 예시 ---")
print(conversation_kg_chain.invoke("제 이름은 에타이고, 제 친구 이름은 세타입니다. 우리는 서울에 삽니다.")['response'])
# 메모리 (내부 KG): (에타, is named, 에타), (세타, is named, 세타), (에타, has friend, 세타), (에타, lives in, 서울), (세타, lives in, 서울) 등 추출 시도

print(conversation_kg_chain.invoke("에타의 친구는 누구인가요?")['response'])
# AI는 KG를 참조하여 "세타입니다."라고 답변할 수 있음

print("\n최종 메모리 내용 (KG - 요약된 지식 그래프):")
print(memory_kg.load_memory_variables({"input": "아무거나"})['kg_history']) # 현재 입력과 무관하게 전체 KG 요약 요청
```

-----

## 4\. 메모리 사용 시 고려사항

  * **토큰 제한 및 비용**: 특히 버퍼 기반 메모리는 대화가 길어질수록 많은 토큰을 소모하여 비용이 증가하고 LLM의 컨텍스트 윈도우를 초과할 수 있습니다.
  * **정보 손실 가능성**: 요약 메모리나 윈도우 메모리는 정보 손실의 위험이 있습니다. 애플리케이션의 특성에 따라 허용 가능한 손실 수준을 고려해야 합니다.
  * **애플리케이션의 특성**:
      * 짧고 간단한 대화: `ConversationBufferMemory`
      * 긴 대화지만 최근 맥락이 중요: `ConversationBufferWindowMemory`
      * 토큰 수 제어가 중요: `ConversationTokenBufferMemory`
      * 매우 긴 대화의 핵심 맥락 유지: `ConversationSummaryMemory`, `ConversationSummaryBufferMemory`
      * 방대한 과거 정보에서 특정 내용 검색: `VectorStoreRetrieverMemory`
  * **구현 복잡도**: 고급 메모리일수록 설정과 관리가 복잡해집니다.
  * **개인정보보호**: 메모리에 민감한 정보가 저장될 수 있으므로, 보안 및 개인정보보호 규정을 준수해야 합니다.
