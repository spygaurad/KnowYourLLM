# from typing import Literal

# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import StateGraph
# from langgraph.prebuilt import tools_condition

# builder = StateGraph(State)


# def user_info(state: State):
#     return {"user_info": fetch_user_flight_information.invoke({})}


# builder.add_node("fetch_user_info", user_info)
# builder.add_edge(START, "fetch_user_info")

# # Flight booking assistant
# builder.add_node(
#     "enter_update_flight",
#     create_entry_node("Flight Updates & Booking Assistant", "update_flight"),
# )
# builder.add_node("update_flight", Assistant(update_flight_runnable))
# builder.add_edge("enter_update_flight", "update_flight")
# builder.add_node(
#     "update_flight_sensitive_tools",
#     create_tool_node_with_fallback(update_flight_sensitive_tools),
# )
# builder.add_node(
#     "update_flight_safe_tools",
#     create_tool_node_with_fallback(update_flight_safe_tools),
# )


# def route_update_flight(
#     state: State,
# ):
#     route = tools_condition(state)
#     if route == END:
#         return END
#     tool_calls = state["messages"][-1].tool_calls
#     did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
#     if did_cancel:
#         return "leave_skill"
#     safe_toolnames = [t.name for t in update_flight_safe_tools]
#     if all(tc["name"] in safe_toolnames for tc in tool_calls):
#         return "update_flight_safe_tools"
#     return "update_flight_sensitive_tools"


# builder.add_edge("update_flight_sensitive_tools", "update_flight")
# builder.add_edge("update_flight_safe_tools", "update_flight")
# builder.add_conditional_edges(
#     "update_flight",
#     route_update_flight,
#     ["update_flight_sensitive_tools", "update_flight_safe_tools", "leave_skill", END],
# )


# # This node will be shared for exiting all specialized assistants
# def pop_dialog_state(state: State) -> dict:
#     """Pop the dialog stack and return to the main assistant.

#     This lets the full graph explicitly track the dialog flow and delegate control
#     to specific sub-graphs.
#     """
#     messages = []
#     if state["messages"][-1].tool_calls:
#         # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
#         messages.append(
#             ToolMessage(
#                 content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
#                 tool_call_id=state["messages"][-1].tool_calls[0]["id"],
#             )
#         )
#     return {
#         "dialog_state": "pop",
#         "messages": messages,
#     }


# builder.add_node("leave_skill", pop_dialog_state)
# builder.add_edge("leave_skill", "primary_assistant")