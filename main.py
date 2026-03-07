from graph import build_graph
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

config = RunnableConfig(
    configurable={
        "thread_id": "research-001",
    }
)


inputs = {"messages": [HumanMessage(content="Domain adaptation in clinical drug")]}
graph = build_graph()
output = graph.invoke(inputs, config)

print(output["messages"][-1].content)

# 그래프 상태 스냅샷 생성
snapshot = graph.get_state(config)

# 다음 스냅샷 상태 접근
snapshot.next

graph.update_state(
    config,
    {
        "query_approved": True,
        "ask_human": False,
        "messages": [HumanMessage(content="APPROVE")],
    },
)

for event in graph.stream(None, config, stream_mode="values"):
    if "messages" in event:
        print("==>", event["messages"][-1].content)

output2 = graph.invoke(None, config)
print(output2["messages"][-1].content)