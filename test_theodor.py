import asyncio
from agents.theodor_agent.agent import initialize_agent
from langchain_core.messages import HumanMessage
from langgraph.types import Command

async def test_agent():
    print("Initializing agent...")
    graph = initialize_agent()
    try:
        print(graph.get_graph().draw_ascii())
    except:
        print("Could not draw graph")
    
    config = {"configurable": {"thread_id": "debug_thread_4_split"}}
    
    print("\n--- Step 1: Start Session ---")
    inputs = {"messages": [HumanMessage(content="Привет, давай начнем")]}
    
    # 1. Start
    print(">> Sending 'Start'...")
    async for event in graph.astream(inputs, config=config, stream_mode="values"):
        pass
    
    # Check if interrupted
    state = graph.get_state(config)
    print(f"After start - interrupted: {state.values.get('__interrupt__')}")
    
    # 2. Choose Option A
    print("\n>> Sending 'Option A'...")
    inputs = {"messages": [HumanMessage(content="Выбираю вариант A")]}
    async for event in graph.astream(inputs, config=config, stream_mode="values"):
        if "messages" in event:
            print(f"Bot: {ascii(event['messages'][-1].content[:50])}...")

    # Check if interrupted (should be waiting for confirmation)
    state = graph.get_state(config)
    print(f"\nAfter Option A - Next tasks: {state.next}")
    if hasattr(state.values, '__interrupt__'):
        print(f"Interrupt info: {state.values.get('__interrupt__')}")
    
    # 3. Confirm using Command(resume=...) with natural language
    print("\n>> Sending 'Everything is great, let's continue' via Command(resume=...)...")
    async for event in graph.astream(Command(resume="Все отлично, идем дальше"), config=config, stream_mode="values"):
        if "messages" in event:
            last_msg = event['messages'][-1]
            print(f"Bot ({type(last_msg).__name__}): {ascii(last_msg.content[:50])}...")
            
    # Check final state
    state = graph.get_state(config)
    print(f"\n=== FINAL STATE ===")
    print(f"Current Step Index: {state.values.get('current_step_index')}")
    print(f"Next tasks: {state.next}")
        
    if state.values.get('current_step_index') == 0:
        print("FAIL: Still on Step 0")
    else:
        print("SUCCESS: Advanced to Step 1")

if __name__ == "__main__":
    asyncio.run(test_agent())
