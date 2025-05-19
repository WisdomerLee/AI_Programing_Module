LangGraph는 기존 Langchain (특히 LCEL)의 한계를 보완하고, 더 복잡하고 동적인 LLM 애플리케이션을 구축할 수 있도록 설계된 라이브러리입니다.

-----

# 1\. LangGraph 도입 배경: Langchain (LCEL)의 어떤 부분을 보강했는가?

Langchain Expression Language (LCEL)은 선언적으로 체인을 구성하고 실행하는 데 매우 강력하며, 스트리밍, 비동기, 병렬 처리 등을 훌륭하게 지원합니다.
LCEL은 기본적으로 **DAG(Directed Acyclic Graph, 방향성 비순환 그래프)** 형태로 작동합니다.
즉, 데이터 흐름이 한 방향으로 진행되며 명시적인 순환(루프)이나 복잡한 조건부 분기 후 다시 합쳐지는 등의 제어 흐름을 직접 표현하기에는 한계가 있었습니다.

**기존 방식(LCEL 포함)의 한계점:**

1.  **순환(Cycles) 및 루프 표현의 어려움**:

      * 많은 실제 LLM 애플리케이션, 특히 에이전트(Agent)는 "생각 -\> 행동 -\> 관찰"과 같은 **순환적인 루프**를 필요로 합니다. LLM의 판단에 따라 여러 번 도구를 호출하거나, 사용자 피드백을 받아 다시 특정 단계를 실행하는 등의 로직을 LCEL만으로 우아하게 표현하기 어려웠습니다.
      * 기존 `AgentExecutor`는 이러한 루프를 내부적으로 처리했지만, 개발자가 이 루프를 세밀하게 제어하거나 커스터마이징하는 데는 제약이 있었습니다.

2.  **복잡한 제어 흐름(Control Flow) 관리**:

      * 상황에 따라 실행 경로가 여러 갈래로 나뉘었다가 다시 합쳐지거나, 특정 조건 만족 시까지 반복하는 등의 복잡한 제어 흐름을 LCEL 파이프라인만으로 구현하기에는 부자연스러웠습니다.

3.  **명시적인 상태 관리(State Management)의 부재**:

      * 긴 대화나 여러 단계의 작업을 수행하는 에이전트는 현재까지의 진행 상황, 중간 결과 등의 "상태(state)"를 명시적으로 관리해야 합니다. LCEL에서는 이러한 상태를 외부에서 관리하거나 메모리 모듈을 통해 간접적으로 다루어야 했습니다.

**LangGraph가 보강하는 부분:**

LangGraph는 이러한 한계를 극복하기 위해 도입되었습니다. 핵심은 **상태 머신(State Machine) 또는 그래프(Graph) 이론**에 기반하여, **상태(State)를 중심으로 노드(Node)와 엣지(Edge)를 정의하여 순환을 포함한 복잡한 실행 흐름을 명시적으로 구축**할 수 있도록 하는 것입니다.

-----

# 2\. LangGraph와 LCEL의 주요 차이점

| 특징             | LCEL (Langchain Expression Language)                                  | LangGraph                                                              |
| ---------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **기본 구조** | DAG (방향성 비순환 그래프), 선형적인 파이프라인                        | 상태 머신, 순환 그래프 (Cyclical Graph)                                  |
| **주요 용도** | 비교적 간단한 순차/병렬 처리, RAG 파이프라인, 간단한 에이전트 로직의 일부 | 복잡한 에이전트, 멀티 에이전트 시스템, 인간 참여 루프, 복잡한 조건부/반복 로직 |
| **제어 흐름** | 주로 데이터 흐름에 초점                                                | 노드와 엣지를 통해 명시적인 제어 흐름 정의 (조건부 엣지 등)             |
| **상태 관리** | 주로 외부 또는 메모리 모듈을 통해 간접적 관리                          | 그래프 전체의 "상태(State)"를 명시적으로 정의하고 각 노드가 이를 업데이트 |
| **순환/루프** | 직접적인 표현 어려움                                                 | 명시적인 순환 및 루프 구성 가능                                         |
| **유연성** | 선언적이고 간결하지만, 복잡한 제어에는 한계                             | 매우 유연하여 거의 모든 종류의 LLM 기반 워크플로우 구축 가능           |
| **디버깅/추적** | LangSmith 등으로 추적 용이                                             | LangSmith 등으로 각 노드 실행 및 상태 변화 추적 용이                    |

**핵심 차이 요약**: LCEL이 데이터 흐름을 연결하는 데 중점을 둔다면, LangGraph는 **상태 변화와 제어 흐름을 중심으로** LLM 애플리케이션을 구축합니다. LCEL은 LangGraph 내의 노드를 구성하는 데 여전히 사용될 수 있습니다. 즉, LangGraph는 LCEL을 포함하여 더 큰 그림을 그리는 도구라고 볼 수 있습니다.

-----

# 3\. LangGraph 모듈의 주요 개념 및 구성 요소

LangGraph는 몇 가지 핵심 개념을 중심으로 작동합니다.

1.  **`Graph`**:

      * 전체 워크플로우를 나타내는 객체입니다. 여기에 노드와 엣지를 추가하여 실행 흐름을 정의합니다.
      * 가장 일반적으로 `StatefulGraph`를 사용하며, 이는 그래프 실행 전반에 걸쳐 상태를 유지합니다.

2.  **`State` (상태)**:

      * 그래프 실행 중에 유지되고 각 노드에 의해 업데이트될 수 있는 데이터의 스키마입니다.
      * 일반적으로 Python의 `TypedDict`나 Pydantic `BaseModel`을 사용하여 정의합니다.
      * 그래프의 "메모리" 역할을 하며, 모든 노드는 이 상태의 일부 또는 전체에 접근하고 수정할 수 있습니다.
      * 예시: 사용자의 질문, 검색된 문서, LLM의 답변, 현재까지의 시도 횟수 등을 상태에 포함할 수 있습니다.

3.  **`Nodes` (노드)**:

      * 그래프 내에서 특정 작업을 수행하는 단위입니다. 일반 Python 함수나 LCEL Runnable 객체가 될 수 있습니다.
      * 각 노드는 현재 **상태(State)** 객체를 입력으로 받고, 작업을 수행한 후 변경된 상태(또는 상태의 일부)를 반환합니다.
      * **일반 노드**: 특정 비즈니스 로직, LLM 호출, 도구 사용 등을 수행합니다.
      * **(조건부 엣지를 위한) 라우팅 노드**: 다음으로 어떤 노드로 이동할지 결정하는 로직을 담고 있는 노드입니다. 이 노드의 반환값에 따라 조건부 엣지가 특정 경로를 선택합니다.

4.  **`Edges` (엣지)**:

      * 노드 간의 연결을 나타내며, 제어 흐름을 정의합니다.
      * **일반 엣지 (`add_edge(start_node_name, end_node_name)`)**: `start_node_name` 노드가 완료된 후 항상 `end_node_name` 노드를 실행합니다.
      * **조건부 엣지 (`add_conditional_edges(source_node_name, path_function, path_map)`)**:
          * `source_node_name`: 조건부 분기가 시작되는 노드 (보통 라우팅 노드).
          * `path_function`: `source_node_name`의 실행 결과를 입력으로 받아, 다음으로 실행할 노드의 이름(또는 특수 값 `END`)을 반환하는 함수.
          * `path_map`: `path_function`이 반환할 수 있는 각 값에 대해 실제로 연결될 다음 노드의 이름을 매핑한 딕셔너리.
      * **`set_entry_point(node_name)`**: 그래프 실행 시 가장 먼저 실행될 노드를 지정합니다.
      * **`set_finish_point(node_name)`**: (최신 버전에서는 `END`를 사용하는 것이 일반적) 특정 노드가 완료되면 그래프 실행을 종료하도록 지정할 수 있습니다.

5.  **`compile()`**:

      * 정의된 노드와 엣지를 바탕으로 실행 가능한 `CompiledGraph` (Runnable) 객체를 생성합니다.
      * 이 컴파일된 그래프는 LCEL의 `Runnable` 인터페이스를 따르므로 `invoke`, `stream`, `batch` 등의 메서드를 사용할 수 있습니다.

6.  **`END`**:

      * 그래프 실행을 종료시키는 특별한 노드 이름입니다. 조건부 엣지의 `path_function`이 `END`를 반환하면 해당 경로에서 그래프 실행이 멈춥니다.

-----

# 4\. LangGraph 코드 예시

여기서는 두 가지 예시를 보여드리겠습니다:

1.  매우 간단한 카운터 예시 (상태 업데이트와 루프)
2.  기본적인 ReAct 스타일 에이전트의 골격

**예시 1: 간단한 카운터 (루프 및 상태 업데이트)**

```python
from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END

# 1. 상태 정의
class CounterState(TypedDict):
    count: int
    max_count: int

# 2. 노드 함수 정의
def increment_counter(state: CounterState) -> CounterState:
    print(f"현재 카운트: {state['count']}")
    return {"count": state['count'] + 1}

def check_condition(state: CounterState) -> str:
    if state['count'] >= state['max_count']:
        print("최대 카운트 도달. 종료합니다.")
        return "finish"
    else:
        print("카운트 계속합니다.")
        return "continue_counting"

# 3. 그래프 생성
workflow = StateGraph(CounterState)

# 노드 추가
workflow.add_node("incrementer", increment_counter)
workflow.add_node("condition_checker", check_condition)

# 엣지 설정
workflow.set_entry_point("incrementer") # 시작점
workflow.add_edge("incrementer", "condition_checker") # incrementer 다음에는 항상 condition_checker 실행

# 조건부 엣지 설정
workflow.add_conditional_edges(
    "condition_checker", # 조건 분기 시작 노드
    check_condition,     # 실제로는 condition_checker 노드의 출력을 사용하지만, 여기서는 명시
                         # 이 노드의 반환값("finish" 또는 "continue_counting")에 따라 경로 결정
    {
        "finish": END,  # "finish"를 반환하면 그래프 종료
        "continue_counting": "incrementer" # "continue_counting"을 반환하면 다시 incrementer로 (루프)
    }
)

# 4. 그래프 컴파일
app = workflow.compile()

print("--- LangGraph 카운터 예시 (max_count=3) ---")
# 초기 상태와 함께 실행
initial_state = {"count": 0, "max_count": 3}
for event in app.stream(initial_state):
    print(event) # 각 노드 실행 결과 및 상태 변화 스트리밍

# 최종 상태 확인
final_state = app.invoke(initial_state)
print(f"최종 상태: {final_state}")
```

**예시 2: 간단한 ReAct 스타일 에이전트의 골격 (LangGraph 사용)**

이 예시는 도구 사용까지는 구현하지 않고, LLM 호출과 조건부 분기 흐름만 보여줍니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver # 상태 저장을 위한 체크포인터 (선택적)

# 상태 정의
class AgentState(TypedDict):
    input: str                                  # 사용자 입력
    agent_outcome: AIMessage | None             # LLM의 결정 (도구 호출 또는 최종 답변)
    intermediate_steps: Annotated[Sequence[tuple[AIMessage, str]], operator.add] # (도구 호출 메시지, 도구 결과 문자열) 리스트
    final_answer: str | None                    # 최종 답변

# LLM (에이전트의 두뇌 역할)
# 실제로는 ReAct 프롬프트를 사용하고, 도구 정보를 전달해야 함
llm = ChatOpenAI(model="gpt-4o", temperature=0) # 예시로 gpt-4o 사용

# 노드 함수 정의
def call_agent_llm(state: AgentState) -> dict:
    print("--- 에이전트 LLM 호출 ---")
    # 실제 에이전트 프롬프트는 더 복잡함 (도구 설명, 이전 단계 등 포함)
    messages = [HumanMessage(content=state['input'])]
    if state.get('intermediate_steps'): # 이전 도구 사용 결과가 있다면 함께 전달
        for tool_call_msg, tool_output in state['intermediate_steps']:
            messages.append(tool_call_msg) # 이전 LLM의 tool_calls
            # ToolMessage는 tool_call_id와 content(도구 실행 결과)를 가짐
            # 여기서는 단순화를 위해 문자열로 추가한다고 가정
            messages.append(ToolMessage(content=str(tool_output), tool_call_id="dummy_id"))


    response = llm.invoke(messages) # LLM 호출
    # response는 AIMessage 객체. tool_calls 속성이 있거나, content가 최종 답변일 수 있음
    return {"agent_outcome": response}

def run_tool(state: AgentState) -> dict:
    print("--- 도구 실행 (시뮬레이션) ---")
    agent_action = state['agent_outcome']
    tool_outputs = []
    # 실제로는 agent_action.tool_calls를 순회하며 도구 실행
    if agent_action and agent_action.tool_calls:
        for tool_call in agent_action.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            print(f"도구 '{tool_name}' 실행, 인자: {tool_args}")
            # 여기서 실제 도구 실행 로직 (예: search.run(tool_args['query']))
            # 지금은 더미 결과 반환
            tool_result = f"도구 '{tool_name}' 실행 결과 (더미)"
            tool_outputs.append((agent_action, tool_result)) # (도구 호출 메시지, 도구 결과)
        return {"intermediate_steps": tool_outputs}
    return {}


def should_continue(state: AgentState) -> str:
    print("--- 계속 여부 결정 ---")
    if state['agent_outcome'] and state['agent_outcome'].tool_calls:
        # LLM이 도구를 사용하라고 결정했으면
        return "continue_to_tool"
    else:
        # LLM이 최종 답변을 생성했으면
        # 실제로는 여기서 agent_outcome.content를 final_answer로 옮기는 노드가 더 필요할 수 있음
        print("최종 답변 단계로 이동")
        return "finish"

# 그래프 구성
agent_workflow = StateGraph(AgentState)

agent_workflow.add_node("agent_llm", call_agent_llm)
agent_workflow.add_node("tool_executor", run_tool)
# 최종 답변을 처리하는 노드를 추가할 수 있음 (예: agent_outcome.content를 final_answer로)

agent_workflow.set_entry_point("agent_llm")

agent_workflow.add_conditional_edges(
    "agent_llm", # agent_llm 노드 실행 후
    should_continue, # should_continue 함수의 반환값에 따라 분기
    {
        "continue_to_tool": "tool_executor",
        "finish": END
    }
)
# tool_executor 실행 후에는 다시 agent_llm으로 돌아가서 도구 결과를 보고 다음 행동 결정 (루프)
agent_workflow.add_edge("tool_executor", "agent_llm")


# 메모리(체크포인터) 설정: 실행 상태를 저장하고 이어갈 수 있게 함 (선택 사항)
# memory = SqliteSaver.from_conn_string(":memory:") # 인메모리 SQLite
# app_agent = agent_workflow.compile(checkpointer=memory)
app_agent = agent_workflow.compile()


print("\n--- LangGraph 에이전트 골격 예시 ---")
# 입력
initial_agent_state = {"input": "뉴욕의 현재 날씨는 어떤가요?", "intermediate_steps": []}

# 실행 (스트리밍)
# config는 쓰레드별로 고유한 ID를 제공하여 대화 상태를 구분할 수 있게 함 (체크포인터 사용 시)
# config = {"configurable": {"thread_id": "user123"}}
for event in app_agent.stream(initial_agent_state): #, config=config):
    print(f"이벤트 타입: {event.get('event', 'N/A')}")
    for key, value in event.items():
        if key != 'event': # 노드 이름이 키
            print(f"  노드 '{key}': {value}")
    print("----")

# 최종 결과
# final_agent_response = app_agent.invoke(initial_agent_state) #, config=config)
# print(f"\n최종 에이전트 상태: {final_agent_response}")
```

위 에이전트 예시는 매우 간략화된 골격이며, 실제로는 에이전트 프롬프트, 도구 정의 및 실행, 오류 처리 등이 훨씬 정교하게 구현되어야 합니다.

-----

# 5\. LangGraph의 장점 요약

  * **최고 수준의 유연성 및 제어력**: 거의 모든 종류의 복잡한 LLM 워크플로우를 상태, 노드, 엣지로 명시적으로 모델링하고 제어할 수 있습니다.
  * **순환 및 루프 지원**: 에이전트의 반복적인 "생각-행동-관찰" 루프, 사용자 피드백 루프 등을 자연스럽게 구현합니다.
  * **명시적인 상태 관리**: 그래프 전체의 상태를 중앙에서 관리하고 각 노드가 이를 업데이트하므로, 복잡한 상호작용에서 상태 추적이 용이합니다.
  * **복잡한 에이전트 구축 용이**: 멀티 에이전트 시스템, 자가 수정(self-correcting) 에이전트 등 고급 에이전트 아키텍처를 구축하는 데 적합합니다.
  * **인간 참여 루프 (Human-in-the-loop)**: 특정 단계에서 인간의 검토나 승인을 받는 워크플로우를 쉽게 통합할 수 있습니다.
  * **디버깅 및 시각화 향상**: 상태와 제어 흐름이 명확하므로 LangSmith와 같은 도구를 통해 실행 과정을 시각화하고 디버깅하기 좋습니다.
