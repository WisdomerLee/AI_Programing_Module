**여러 구성 요소(모델, 프롬프트, 다른 체인, 메모리, 데이터베이스 등)를 논리적인 순서나 방식으로 연결하여 특정 작업을 수행하는 실행 단위**

-----

# 1\. `LLMChain` (기본 LLM 체인)

가장 기본적이고 핵심적인 체인입니다. 프롬프트 템플릿과 LLM(또는 챗 모델)을 결합하여 사용자 입력을 받아 LLM에 전달하고 그 응답을 반환합니다.

**특징:**

  * 단일 LLM 호출을 위한 간단한 구성.
  * 프롬프트 템플릿을 통해 동적인 입력 처리.
  * LLM 또는 챗 모델 모두 사용 가능.

**주요 구성 요소:**

  * `llm`: LLM 객체 (예: `OpenAI`) 또는 챗 모델 객체 (예: `ChatOpenAI`).
  * `prompt`: `PromptTemplate` 또는 `ChatPromptTemplate` 객체.
  * `output_key` (선택 사항): 출력 딕셔셔너리에서 LLM 응답에 해당하는 키 이름 (기본값은 보통 "text" 또는 모델에 따라 다름).

**Python 코드 예시:**

```python
import os
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain

# API 키 설정 (실제 키로 대체하거나 환경 변수로 설정하세요)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 예시 1: LLM과 PromptTemplate 사용
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7)
prompt_llm = PromptTemplate.from_template(
    "다음 주제에 대해 짧은 시를 써주세요: {topic}"
)
llm_chain = LLMChain(llm=llm, prompt=prompt_llm, output_key="poem") # output_key 지정

response_llm_chain = llm_chain.invoke({"topic": "밤하늘의 별"})
print(f"--- LLMChain (LLM) 결과 ---")
print(f"주제: 밤하늘의 별")
print(f"생성된 시:\n{response_llm_chain['poem']}") # output_key로 접근

# 예시 2: Chat Model과 ChatPromptTemplate 사용
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
prompt_chat = ChatPromptTemplate.from_messages([
    ("system", "당신은 사용자의 질문에 친절하게 답변하는 AI입니다."),
    ("human", "{question}")
])
chat_chain = LLMChain(llm=chat_model, prompt=prompt_chat, output_key="answer")

response_chat_chain = chat_chain.invoke({"question": "Langchain의 주요 장점은 무엇인가요?"})
print(f"\n--- LLMChain (Chat Model) 결과 ---")
print(f"질문: Langchain의 주요 장점은 무엇인가요?")
print(f"답변:\n{response_chat_chain['answer']}")
```

-----

# 2\. Sequential Chains (순차적 체인)

여러 체인을 순서대로 연결하여 실행합니다. 한 체인의 출력이 다음 체인의 입력으로 사용될 수 있습니다.

## 2.1. `SimpleSequentialChain`

  * **특징**: 각 단계가 단일 문자열 입력을 받고 단일 문자열 출력을 반환하는 가장 간단한 형태의 순차 체인입니다. 이전 단계의 출력이 그대로 다음 단계의 입력으로 전달됩니다.
  * **주요 구성 요소**: `chains` (순서대로 실행할 체인 객체들의 리스트).

**Python 코드 예시:**

```python
from langchain.chains import SimpleSequentialChain

# 첫 번째 체인: 주제에 대한 질문 생성
llm_q_generator = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7)
prompt_q = PromptTemplate.from_template("'{topic}'에 대한 흥미로운 질문 하나를 만들어주세요.")
chain_one = LLMChain(llm=llm_q_generator, prompt=prompt_q)

# 두 번째 체인: 생성된 질문에 대한 답변
llm_a_generator = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7)
prompt_a = PromptTemplate.from_template("다음 질문에 간략히 답해주세요:\n{question}") # SimpleSequentialChain은 이전 체인의 출력을 'question'으로 받음 (기본값)
chain_two = LLMChain(llm=llm_a_generator, prompt=prompt_a)

# SimpleSequentialChain 구성
# chain_one의 출력이 chain_two의 입력으로 자동으로 전달됨
simple_sequential_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)

topic_input = "양자역학의 신비"
result_simple_seq = simple_sequential_chain.invoke(topic_input)
print(f"\n--- SimpleSequentialChain 결과 ---")
print(f"입력 주제: {topic_input}")
print(f"최종 결과:\n{result_simple_seq['output']}") # SimpleSequentialChain의 기본 출력 키는 'output'
```

## 2.2. `SequentialChain`

  * **특징**: `SimpleSequentialChain`보다 더 유연하며, 여러 입력과 출력을 다룰 수 있습니다. 각 체인의 입력/출력 변수를 명시적으로 매핑할 수 있습니다.
  * **주요 구성 요소**:
      * `chains`: 순서대로 실행할 체인 객체들의 리스트.
      * `input_variables`: 전체 시퀀스의 초기 입력 변수 이름들.
      * `output_variables`: 전체 시퀀스의 최종 출력 변수 이름들.
      * `memory` (선택 사항): 대화 기록 등을 저장할 메모리 객체.

**Python 코드 예시:**

```python
from langchain.chains import SequentialChain

# 체인 1: 소설 제목과 주인공 이름을 받아 줄거리를 생성
llm_plot = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.8)
prompt_plot = PromptTemplate.from_template(
    "소설 제목 '{title}'과 주인공 '{character_name}'에 대한 간략한 줄거리를 작성해주세요."
)
chain_plot = LLMChain(llm=llm_plot, prompt=prompt_plot, output_key="plot_summary") # 출력 키: plot_summary

# 체인 2: 생성된 줄거리를 평가
llm_review = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.5)
prompt_review = PromptTemplate.from_template(
    "다음 줄거리에 대한 한 줄 평을 남겨주세요:\n{plot_summary}" # 이전 체인의 'plot_summary'를 입력으로 받음
)
chain_review = LLMChain(llm=llm_review, prompt=prompt_review, output_key="review_comment") # 출력 키: review_comment

# SequentialChain 구성
sequential_chain = SequentialChain(
    chains=[chain_plot, chain_review],
    input_variables=["title", "character_name"], # 초기 입력
    output_variables=["plot_summary", "review_comment"], # 최종 출력할 변수들
    verbose=True
)

story_inputs = {"title": "잃어버린 시간의 여행자", "character_name": "카이"}
result_seq = sequential_chain.invoke(story_inputs)
print(f"\n--- SequentialChain 결과 ---")
print(f"입력: {story_inputs}")
print(f"줄거리:\n{result_seq['plot_summary']}")
print(f"한 줄 평:\n{result_seq['review_comment']}")
```

-----

# 3\. `RouterChain` (라우터 체인)

사용자 입력에 따라 여러 대상 체인 중 어떤 체인을 실행할지 동적으로 결정(라우팅)합니다.

**특징:**

  * 입력의 내용이나 의도에 따라 다른 처리를 분기할 때 유용.
  * LLM을 사용하여 라우팅 결정을 내리거나, 임베딩 유사도 기반으로 라우팅 가능.

**`LLMRouterChain` 예시:**

  * **구성 요소**:
      * `destination_chains`: 라우팅될 대상 체인들에 대한 정보 (이름, 설명, 실제 체인).
      * `default_chain`: 어떤 대상에도 해당하지 않을 때 실행될 기본 체인.
      * `router_llm`: 라우팅 결정을 내릴 LLM.
      * `router_prompt`: 라우팅 결정을 위한 프롬프트.

**Python 코드 예시:**

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# 대상 체인들 정의
physics_template = """당신은 물리학 전문가입니다. 다음 질문에 답해주세요:
질문: {input}
답변:"""
physics_prompt = PromptTemplate(template=physics_template, input_variables=["input"])
physics_chain = LLMChain(llm=OpenAI(), prompt=physics_prompt)

math_template = """당신은 수학 전문가입니다. 다음 질문에 답해주세요:
질문: {input}
답변:"""
math_prompt = PromptTemplate(template=math_template, input_variables=["input"])
math_chain = LLMChain(llm=OpenAI(), prompt=math_prompt)

# 각 대상 체인에 대한 정보 (라우터가 참고할 설명)
prompt_infos = [
    {
        "name": "physics",
        "description": "물리학 관련 질문에 답변하기 좋습니다.",
        "prompt_template": physics_template # 실제 프롬프트 템플릿 문자열
    },
    {
        "name": "math",
        "description": "수학 문제 해결이나 수학 개념 질문에 답변하기 좋습니다.",
        "prompt_template": math_template
    },
]

# 라우팅을 결정할 LLM
router_llm = OpenAI(temperature=0)

# 라우팅 프롬프트 (라우터 LLM이 이 프롬프트를 보고 어떤 destination으로 보낼지 결정)
# MultiPromptChain.create_router_prompt_from_prompts() 를 사용하면 쉽게 생성 가능
# 여기서는 예시로 직접 구성 (실제로는 더 정교해야 함)
destinations_str = "\n".join([f"{p['name']}: {p['description']}" for p in prompt_infos])
router_template = f"""주어진 사용자 질문에 가장 적합한 전문가를 선택하세요.
선택지는 다음과 같습니다:
{destinations_str}

사용자 질문:
{{input}}

선택된 전문가 (위 선택지 중 하나의 이름만 정확히 출력):"""
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(), # LLM의 출력을 라우팅 결정 형식으로 파싱
)

# 라우터 체인 생성
router_chain = LLMRouterChain.from_llm(router_llm, router_prompt)

# MultiPromptChain: 라우터와 대상 체인들을 연결
# default_chain은 적절한 대상을 찾지 못했을 때 사용
default_chain = LLMChain(llm=OpenAI(), prompt=PromptTemplate.from_template("일반적인 질문입니다: {input}"))
multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains={
        "physics": physics_chain,
        "math": math_chain,
    },
    default_chain=default_chain,
    verbose=True,
)

print(f"\n--- RouterChain (MultiPromptChain) 결과 ---")
print(f"질문: 빛의 이중성이란 무엇인가요?")
print(multi_prompt_chain.invoke("빛의 이중성이란 무엇인가요?")) # physics로 라우팅될 것으로 예상

print(f"\n질문: 1부터 100까지 더하면 얼마인가요?")
print(multi_prompt_chain.invoke("1부터 100까지 더하면 얼마인가요?")) # math로 라우팅될 것으로 예상
```

-----

# 4\. `TransformChain` (변환 체인)

입력 데이터를 특정 함수를 사용하여 변환하거나, LLM의 출력을 후처리하는 데 사용됩니다.

**특징:**

  * 데이터 전처리 및 후처리를 파이프라인에 통합.
  * 사용자 정의 Python 함수를 체인의 일부로 실행.

**주요 구성 요소:**

  * `input_variables`: 변환 함수에 전달될 입력 변수 이름들.
  * `output_variables`: 변환 함수가 반환하는 값에 대한 출력 변수 이름들.
  * `transform`: 실제 변환을 수행하는 Python 함수.

**Python 코드 예시:**

```python
from langchain.chains import TransformChain

# LLM이 생성한 쉼표로 구분된 목록 문자열을 Python 리스트로 변환하는 함수
def parse_comma_separated_list(text_input: str) -> dict:
    items = [item.strip() for item in text_input.split(',') if item.strip()]
    return {"parsed_list": items}

# 변환 체인 생성
list_parser_chain = TransformChain(
    input_variables=["raw_list_string"],
    output_variables=["parsed_list"],
    transform=parse_comma_separated_list
)

# LLMChain과 TransformChain을 SequentialChain으로 연결
llm_list_generator = OpenAI(model_name="gpt-3.5-turbo-instruct")
prompt_list = PromptTemplate.from_template("'{category}'에 해당하는 아이템 3가지를 쉼표로 구분해서 알려줘.")
chain_generate_list = LLMChain(llm=llm_list_generator, prompt=prompt_list, output_key="raw_list_string")

# 최종 파이프라인
pipeline_with_transform = SequentialChain(
    chains=[chain_generate_list, list_parser_chain],
    input_variables=["category"],
    output_variables=["raw_list_string", "parsed_list"], # 모든 중간/최종 결과 확인 가능
    verbose=True
)

result_transform = pipeline_with_transform.invoke({"category": "과일"})
print(f"\n--- TransformChain 결과 ---")
print(f"입력 카테고리: 과일")
print(f"LLM 생성 문자열: {result_transform['raw_list_string']}")
print(f"파싱된 리스트: {result_transform['parsed_list']}")
print(f"파싱된 리스트의 타입: {type(result_transform['parsed_list'])}")
```

-----

# 5\. `RetrievalQA` Chain (검색 증강 생성 체인)

RAG(Retrieval Augmented Generation) 패턴을 구현하는 핵심 체인입니다. 외부 문서 저장소(Vector Store)에서 관련 문서를 검색(Retrieve)하고, 이 문서를 컨텍스트로 활용하여 LLM이 질문에 대한 답변을 생성(Generate)합니다.

**특징:**

  * LLM이 알지 못하는 최신 정보나 특정 도메인 지식에 대해 답변 가능.
  * "Hallucination" (환각, 즉 LLM이 잘못된 정보를 생성하는 현상)을 줄이는 데 도움.

**주요 구성 요소:**

  * `retriever`: 문서를 검색하는 컴포넌트 (예: `FAISS.as_retriever()`).
  * `combine_documents_chain_kwargs` 또는 `chain_type`: 검색된 문서들을 LLM에 어떻게 전달하고 처리할지 정의.
      * `stuff` (기본값): 모든 검색된 문서를 하나의 프롬프트에 넣어 LLM에 전달. (문서가 너무 많으면 토큰 제한 문제 발생 가능)
      * `map_reduce`: 각 문서를 개별적으로 요약/처리하고(Map), 그 결과들을 다시 요약/통합(Reduce).
      * `refine`: 첫 번째 문서로 초기 답변을 생성하고, 다음 문서들을 보면서 답변을 점진적으로 개선.
      * `map_rerank`: 각 문서에 대해 답변을 생성하고, 답변의 신뢰도를 평가하여 가장 좋은 답변을 선택.
  * `llm`: 답변 생성에 사용될 LLM 또는 챗 모델.

**Python 코드 예시 (간단한 인메모리 예시):**

```python
from langchain_community.vectorstores import FAISS # langchain-community 필요
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# 예시 문서들
documents = [
    Document(page_content="Langchain은 LLM을 활용한 애플리케이션 개발을 위한 프레임워크입니다.", metadata={"source": "doc1"}),
    Document(page_content="Langchain의 주요 구성 요소에는 Models, Prompts, Chains, Indexes, Agents, Memory가 있습니다.", metadata={"source": "doc2"}),
    Document(page_content="FAISS는 효율적인 유사도 검색 라이브러리입니다.", metadata={"source": "doc3"}),
    Document(page_content="RetrievalQA 체인은 검색과 생성을 결합합니다.", metadata={"source": "doc4"}),
]

# 임베딩 모델 및 벡터 스토어 생성
embeddings = OpenAIEmbeddings()
try:
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    # RetrievalQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-3.5-turbo-instruct"),
        chain_type="stuff", # 검색된 문서를 처리하는 방식
        retriever=retriever,
        return_source_documents=True # 출처 문서도 함께 반환할지 여부
    )

    query = "Langchain의 주요 구성 요소는 무엇인가요?"
    result_qa = qa_chain.invoke({"query": query}) # 최신 버전에서는 query를 딕셔너리로 전달

    print(f"\n--- RetrievalQA 결과 ---")
    print(f"질문: {query}")
    print(f"답변: {result_qa['result']}")
    print(f"출처 문서:")
    for doc in result_qa['source_documents']:
        print(f"  - {doc.page_content} (출처: {doc.metadata['source']})")

except ImportError:
    print("\nFAISS를 사용하려면 `pip install faiss-cpu` 또는 `pip install faiss-gpu`를 실행해주세요.")
except Exception as e:
    print(f"\nRetrievalQA 예외 발생: {e}. OpenAI API 키 및 환경 설정을 확인해주세요.")


```

-----

# 6\. `ConversationChain` (대화 체인)

대화의 맥락을 유지하기 위해 메모리(Memory)를 사용하는 체인입니다. 이전 대화 내용을 기억하여 보다 자연스러운 대화가 가능하게 합니다.

**특징:**

  * 챗봇과 같이 지속적인 상호작용이 필요한 애플리케이션에 적합.
  * 다양한 종류의 메모리 (예: `ConversationBufferMemory`, `ConversationSummaryMemory`) 사용 가능.

**주요 구성 요소:**

  * `llm`: LLM 또는 챗 모델.
  * `memory`: 대화 기록을 저장하고 관리하는 메모리 객체.
  * `prompt` (선택 사항): 대화 형식을 정의하는 프롬프트 (기본 프롬프트 사용 가능).

**Python 코드 예시:**

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 대화 체인 생성 (ConversationBufferMemory 사용)
conversation_llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
memory = ConversationBufferMemory() # 이전 대화 내용을 그대로 저장
conversation_chain = ConversationChain(llm=conversation_llm, memory=memory, verbose=True)

print(f"\n--- ConversationChain 결과 ---")
print("AI: 안녕하세요! 무엇을 도와드릴까요?") # 초기 메시지 (개발자가 설정)

user_input1 = "제 이름은 홍길동입니다."
ai_response1 = conversation_chain.invoke(user_input1)
print(f"나: {user_input1}")
print(f"AI: {ai_response1['response']}") # ConversationChain의 기본 출력 키는 'response'

user_input2 = "제 이름을 기억하시나요?"
ai_response2 = conversation_chain.invoke(user_input2)
print(f"나: {user_input2}")
print(f"AI: {ai_response2['response']}") # 메모리를 통해 이전 대화(홍길동)를 기억하고 답변

# 메모리 내용 확인
print("\n대화 메모리:")
print(memory.buffer)
```

-----
