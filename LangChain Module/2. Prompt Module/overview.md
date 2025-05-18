
LLM(대규모 언어 모델)의 성능은 어떤 프롬프트를 사용하느냐에 따라 크게 달라집니다.
Langchain의 프롬프트 모듈은 **LLM에 전달할 입력을 동적으로 생성하고, 재사용 가능하며, 다양한 요구사항에 맞춰 구조화할 수 있도록 돕는 강력한 도구**입니다.

-----

# 1\. `PromptTemplate` (기본 프롬프트 템플릿)

가장 기본적인 프롬프트 구성 요소로, f-string과 유사하게 작동하는 템플릿 문자열을 사용합니다.

**주요 특징 및 사용법:**

  * **`from_template(template_string)`**: 클래스 메서드를 사용하여 간단히 템플릿 객체를 생성합니다.
  * **`input_variables`**: 템플릿 문자열 내에 `{}`로 감싸진 변수들의 이름을 리스트로 가집니다.
  * **`template`**: 실제 프롬프트 문자열입니다.
  * **`format(**kwargs)`**: 템플릿 내의 변수에 실제 값을 전달하여 최종 프롬프트 문자열을 생성합니다.

**Python 코드 예시:**

```python
from langchain_core.prompts import PromptTemplate

# 1. 간단한 변수 하나를 사용하는 템플릿
prompt_template_product = PromptTemplate.from_template(
    "AI를 활용한 새로운 {product} 아이디어 3가지를 알려주세요."
)

# 변수 값 할당하여 프롬프트 완성
filled_prompt_product = prompt_template_product.format(product="교육 서비스")
print(f"--- 단일 변수 템플릿 ---")
print(filled_prompt_product)
# 출력: AI를 활용한 새로운 교육 서비스 아이디어 3가지를 알려주세요.

# 2. 여러 입력 변수를 사용하는 템플릿
prompt_template_city_info = PromptTemplate(
    input_variables=["city_name", "category"],
    template="{city_name}에서 {category}으로 유명한 장소 2곳을 추천해주세요."
)

filled_prompt_city = prompt_template_city_info.format(city_name="서울", category="맛집")
print(f"\n--- 다중 변수 템플릿 ---")
print(filled_prompt_city)
# 출력: 서울에서 맛집으로 유명한 장소 2곳을 추천해주세요.

# 3. `partial`을 이용한 부분 변수 고정
# 'category' 변수를 '역사 유적지'로 미리 고정
partial_prompt = prompt_template_city_info.partial(category="역사 유적지")

# 나머지 변수('city_name')만 전달하여 프롬프트 완성
filled_partial_prompt = partial_prompt.format(city_name="경주")
print(f"\n--- 부분 변수 고정 템플릿 ---")
print(filled_partial_prompt)
# 출력: 경주에서 역사 유적지로 유명한 장소 2곳을 추천해주세요.

# LLM과 함께 사용 (이전 답변의 llm 객체가 있다고 가정)
# from langchain_openai import OpenAI
# llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
# response = llm.invoke(filled_prompt_product)
# print(f"\n--- LLM 응답 예시 ---")
# print(response)
```

-----

# 2\. `ChatPromptTemplate` (챗 모델용 프롬프트 템플릿)

챗 모델은 단순 문자열이 아닌, 역할(system, human, ai)이 부여된 메시지 객체의 리스트를 입력으로 받습니다. `ChatPromptTemplate`은 이러한 메시지 구조를 쉽게 만들 수 있도록 도와줍니다.

**주요 특징 및 사용법:**

  * **`from_messages(messages)`**: 클래스 메서드를 사용하여 생성하며, `Message` 객체 또는 `(역할_문자열, 내용_템플릿_문자열)` 튜플의 리스트를 인자로 받습니다.
  * **메시지 유형**:
      * `SystemMessagePromptTemplate`: AI의 역할이나 행동 지침을 정의 (보통 대화 시작 시 한 번 사용).
      * `HumanMessagePromptTemplate`: 사용자 입력을 나타내는 메시지 템플릿.
      * `AIMessagePromptTemplate`: AI의 응답을 나타내는 메시지 템플릿 (퓨샷 예제나 대화 기록에 사용).
  * **`format_messages(**kwargs)`**: 변수 값을 전달하여 실제 `Message` 객체들의 리스트를 생성합니다.

**Python 코드 예시:**

```python
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage # 실제 메시지 객체 확인용

# 1. 기본적인 챗 프롬프트 (튜플 형태 사용)
chat_template_basic = ChatPromptTemplate.from_messages([
    ("system", "당신은 사용자의 질문에 간결하고 명확하게 답변하는 AI 어시스턴트입니다."),
    ("human", "{user_question}")
])

formatted_messages_basic = chat_template_basic.format_messages(user_question="Langchain이란 무엇인가요?")
print(f"--- 기본 챗 프롬프트 (튜플 사용) ---")
for msg in formatted_messages_basic:
    print(f"Type: {type(msg)}, Role: {msg.type}, Content: {msg.content}")
# 출력 예시:
# Type: <class 'langchain_core.messages.system.SystemMessage'>, Role: system, Content: 당신은 사용자의 질문에 간결하고 명확하게 답변하는 AI 어시스턴트입니다.
# Type: <class 'langchain_core.messages.human.HumanMessage'>, Role: human, Content: Langchain이란 무엇인가요?

# 2. 메시지 템플릿 객체를 직접 사용
system_prompt = SystemMessagePromptTemplate.from_template(
    "당신은 {input_language}를 {output_language}로 번역하는 전문 번역가입니다."
)
human_prompt = HumanMessagePromptTemplate.from_template(
    "{text_to_translate}"
)
chat_template_translate = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

formatted_messages_translate = chat_template_translate.format_messages(
    input_language="한국어",
    output_language="영어",
    text_to_translate="안녕하세요, 만나서 반갑습니다."
)
print(f"\n--- 메시지 템플릿 객체 사용 ---")
for msg in formatted_messages_translate:
    print(f"Role: {msg.type}, Content: {msg.content}")

# Chat Model과 함께 사용 (이전 답변의 chat 객체가 있다고 가정)
# from langchain_openai import ChatOpenAI
# chat = ChatOpenAI(model_name="gpt-3.5-turbo")
# response_chat = chat.invoke(formatted_messages_basic)
# print(f"\n--- Chat Model 응답 예시 ---")
# print(response_chat.content)
```

-----

# 3\. `FewShotPromptTemplate` (퓨샷 프롬프트 템플릿)

LLM에게 몇 가지 예시(demonstrations)를 함께 제공하여, 원하는 결과물의 형식이나 스타일을 더 명확하게 학습시키는 기법입니다.

**주요 특징 및 사용법:**

  * **`examples`**: LLM에게 보여줄 예시 데이터 목록 (딕셔너리 리스트 형태).
  * **`example_prompt`**: 각 예시를 어떤 형식으로 만들지 정의하는 `PromptTemplate`.
  * **`prefix`**: 예시들 앞에 고정적으로 붙는 설명이나 지시문.
  * **`suffix`**: 예시들 뒤, 그리고 실제 사용자 입력 앞에 붙는 텍스트 (사용자 입력을 받는 부분을 명시).
  * **`input_variables`**: 사용자가 최종적으로 입력할 변수들.

**Python 코드 예시:**

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 1. 예시 데이터 정의
examples = [
    {"input": "행복한", "output": "슬픈"},
    {"input": "뜨거운", "output": "차가운"},
    {"input": "높은", "output": "낮은"}
]

# 2. 각 예시를 포맷할 템플릿 정의
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="입력: {input}\n출력: {output}"
)

# 3. 퓨샷 프롬프트 템플릿 생성
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="다음은 입력 단어의 반대말을 출력하는 예시입니다:",
    suffix="입력: {user_input}\n출력:", # 사용자의 실제 입력 부분
    input_variables=["user_input"] # 사용자가 제공할 변수
)

# 4. 프롬프트 완성
filled_few_shot_prompt = few_shot_prompt.format(user_input="빠른")
print(f"--- 퓨샷 프롬프트 ---")
print(filled_few_shot_prompt)
# 출력 예시:
# 다음은 입력 단어의 반대말을 출력하는 예시입니다:
# 입력: 행복한
# 출력: 슬픈
#
# 입력: 뜨거운
# 출력: 차가운
#
# 입력: 높은
# 출력: 낮은
#
# 입력: 빠른
# 출력:

# LLM과 함께 사용 시, LLM은 '느린'과 유사한 답변을 생성할 확률이 높아집니다.
# response_few_shot = llm.invoke(filled_few_shot_prompt)
# print(f"\n--- 퓨샷 LLM 응답 예시 ---")
# print(response_few_shot)
```

-----

# 4\. `ExampleSelector` (예제 선택기) - 퓨샷 프롬프트와 함께 사용

예시가 매우 많을 경우, 모든 예시를 프롬프트에 포함시키면 토큰 제한을 초과하거나 LLM의 처리 부담을 늘릴 수 있습니다. `ExampleSelector`는 현재 사용자 입력과 가장 관련성이 높거나 적절한 예시들만 동적으로 선택하여 프롬프트에 포함시킵니다.

**주요 ExampleSelector 종류:**

  * **`LengthBasedExampleSelector`**: 예시들의 총 길이가 특정 기준을 넘지 않도록 선택.
  * **`SemanticSimilarityExampleSelector`**: 사용자 입력과 의미적으로 가장 유사한 예시들을 선택 (텍스트 임베딩과 벡터 스토어 필요).
  * **`MaxMarginalRelevanceExampleSelector (MMR)`**: 유사성과 다양성을 함께 고려하여 예시를 선택.

**Python 코드 예시 (`LengthBasedExampleSelector`):**

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

examples = [ # 이전 예시보다 더 많은 예시
    {"input": "행복한", "output": "슬픈"},
    {"input": "뜨거운", "output": "차가운"},
    {"input": "높은", "output": "낮은"},
    {"input": "큰", "output": "작은"},
    {"input": "밝은", "output": "어두운"},
    {"input": "가벼운", "output": "무거운"}
]
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="입력: {input}\n출력: {output}"
)

# 길이 기반 예제 선택기 설정
# 각 예시가 포맷팅된 후의 길이를 고려하여, 총 길이가 max_length를 넘지 않도록 함
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=50  # 예시들의 총 문자열 길이 제한 (대략적인 값)
)

# 퓨샷 프롬프트 템플릿에 ExampleSelector 연결
dynamic_few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector, # 직접 examples를 전달하는 대신 selector를 전달
    example_prompt=example_prompt,
    prefix="다음은 입력 단어의 반대말을 출력하는 예시입니다:",
    suffix="입력: {user_input}\n출력:",
    input_variables=["user_input"]
)

# 프롬프트 완성 (실행 시점에 example_selector가 예시를 선택)
filled_dynamic_prompt = dynamic_few_shot_prompt.format(user_input="조용한")
print(f"--- 동적 퓨샷 프롬프트 (LengthBasedExampleSelector) ---")
print(filled_dynamic_prompt)
# 출력 예시 (max_length에 따라 선택되는 예시 수가 달라짐):
# 다음은 입력 단어의 반대말을 출력하는 예시입니다:
# 입력: 행복한
# 출력: 슬픈
#
# 입력: 뜨거운
# 출력: 차가운
#
# 입력: 조용한
# 출력:
```

**`SemanticSimilarityExampleSelector` 사용 시 참고:**
의미 기반 유사도 선택기는 `OpenAIEmbeddings` 같은 임베딩 모델과 `FAISS`, `Chroma` 같은 벡터 스토어가 필요합니다. 설정이 더 복잡하므로 여기서는 개념만 언급합니다. 사용자 입력과 가장 의미가 비슷한 예시를 찾아주어 더욱 문맥에 맞는 퓨샷 프롬프팅이 가능합니다.

-----

# 5\. `PipelinePromptTemplate` (파이프라인 프롬프트 템플릿)

여러 개의 프롬프트 템플릿을 순차적으로 또는 병렬적으로 조합하여 최종 프롬프트를 구성할 때 사용합니다. 복잡한 프롬프트 로직을 모듈화하고 재사용성을 높일 수 있습니다.

**주요 특징 및 사용법:**

  * `final_prompt`: 최종적으로 사용될 프롬프트 템플릿. 이 템플릿은 파이프라인 내 다른 프롬프트들의 출력을 입력으로 받을 수 있습니다.
  * `pipeline_prompts`: `(이름, PromptTemplate)` 튜플의 리스트. 각 템플릿의 출력은 `final_prompt`에서 `이름`으로 참조됩니다.

**Python 코드 예시:**

```python
from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate

# 1. 개별 프롬프트 템플릿 정의
introduction_template = PromptTemplate.from_template(
    "당신은 {person_role} 전문가입니다. 당신의 임무는 {topic}에 대해 설명하는 것입니다."
)
task_template = PromptTemplate.from_template(
    "특히, 청중이 {audience}일 때 어떤 점을 강조해야 할까요?"
)
style_template = PromptTemplate.from_template(
    "설명은 {style} 스타일로 작성해주세요."
)

# 2. 파이프라인 프롬프트 정의
# 각 프롬프트의 출력이 final_prompt의 입력 변수가 됨
full_template = """{introduction}

{task_detail}

{writing_style}

이제 설명을 시작해주세요:"""

final_prompt_for_pipeline = PromptTemplate.from_template(full_template)

pipeline_prompts_list = [
    ("introduction", introduction_template),
    ("task_detail", task_template),
    ("writing_style", style_template)
]

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=final_prompt_for_pipeline,
    pipeline_prompts=pipeline_prompts_list
)

# 3. 파이프라인 프롬프트 완성
filled_pipeline_prompt = pipeline_prompt.format(
    person_role="양자 물리학",
    topic="슈뢰딩거의 고양이",
    audience="고등학생",
    style="유머러스하고 이해하기 쉬운"
)
print(f"--- 파이프라인 프롬프트 ---")
print(filled_pipeline_prompt)
# 출력 예시:
# 당신은 양자 물리학 전문가입니다. 당신의 임무는 슈뢰딩거의 고양이에 대해 설명하는 것입니다.
#
# 특히, 청중이 고등학생일 때 어떤 점을 강조해야 할까요?
#
# 설명은 유머러스하고 이해하기 쉬운 스타일로 작성해주세요.
#
# 이제 설명을 시작해주세요:
```

-----

# 6\. Output Parsers (출력 파서)와 연동 (간략한 소개)

프롬프트는 LLM에게 원하는 출력의 *형식*을 지시하는 데도 사용될 수 있습니다. `OutputParser`는 LLM의 텍스트 출력을 JSON, 리스트, Pydantic 모델 등 구조화된 데이터로 변환하는 역할을 합니다.

`PromptTemplate`에 `output_parser`를 지정하면, 파서가 제공하는 \*\*형식 지침(`format_instructions`)\*\*을 프롬프트 내에 쉽게 포함시킬 수 있습니다.

**Python 코드 예시 (간단한 `PydanticOutputParser` 연동):**

````python
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.output_parsers.pydantic import PydanticOutputParser # langchain-community 설치 필요

# 1. 원하는 출력 구조를 Pydantic 모델로 정의
class Joke(BaseModel):
    setup: str = Field(description="농담의 설정 부분")
    punchline: str = Field(description="농담의 핵심 부분 (웃음 포인트)")

# 2. Pydantic 모델로부터 출력 파서 생성
parser = PydanticOutputParser(pydantic_object=Joke)

# 3. 프롬프트 템플릿에 format_instructions 포함
# output_parser.get_format_instructions()를 통해 JSON 스키마와 같은 형식 지침을 가져옴
prompt_with_parser = PromptTemplate(
    template="사용자의 요청에 따라 농담을 하나 만들어주세요.\n{format_instructions}\n요청: {user_query}\n",
    input_variables=["user_query"],
    partial_variables={"format_instructions": parser.get_format_instructions()} # 형식 지침 주입
)

# 4. 프롬프트 완성
filled_prompt_with_parser = prompt_with_parser.format(user_query="코딩에 대한 농담")
print(f"--- 출력 파서 연동 프롬프트 ---")
print(filled_prompt_with_parser)
# 출력 예시 (format_instructions 부분에 JSON 스키마가 포함됨):
# 사용자의 요청에 따라 농담을 하나 만들어주세요.
# The output should be formatted as a JSON instance that conforms to the JSON schema below.
#
# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
# the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
#
# Here is the output schema:
# ```json
# {"properties": {"setup": {"title": "Setup", "description": "농담의 설정 부분", "type": "string"}, "punchline": {"title": "Punchline", "description": "농담의 핵심 부분 (웃음 포인트)", "type": "string"}}, "required": ["setup", "punchline"]}
# ```
# 요청: 코딩에 대한 농담

# LLM이 이 프롬프트를 받으면, format_instructions에 따라 JSON 형식으로 농담을 생성하려고 시도합니다.
# 생성된 JSON 문자열은 parser.parse()를 통해 Joke 객체로 변환될 수 있습니다.
````

-----

Langchain의 프롬프트 모듈은 이처럼 다양하고 강력한 기능을 제공하여, LLM과의 상호작용을 정교하게 제어하고 원하는 결과를 얻을 확률을 높여줍니다. 각 구성 요소의 특징을 잘 이해하고 조합하여 활용하면 복잡한 LLM 애플리케이션도 효과적으로 구축할 수 있습니다.
