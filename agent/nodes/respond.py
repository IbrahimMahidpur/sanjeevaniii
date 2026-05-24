"""Respond node for the medical agent.

* If the detected user language is not English, it would invoke the multilingual translator (placeholder).
* Appends the final validated response to the ``messages`` list in the state.
* Returns the updated state.
"""

from typing import Any

async def respond_node(state: Any) -> Any:
    # In a full implementation we would translate here.
    response = state.get("validated_response", "")
    # Append to message history – simple structure
    if "messages" not in state:
        state["messages"] = []
    state["messages"].append({"role": "assistant", "content": response})
    return state
