from langgraph.graph import MessageState
# Whats the weather like in the user selected location


    # Keeps track of the previous and current messages in the conversation
"""

    User is the weather like in this location --> uagent(weather) takes the input and analyzes the data --> RAG agent(LLAMA Index) searches indexed weather data --> langgraph node combines the data and context --> returns the response to the user

"""

class AgentState(MessageState):
    location:str
    weather:str
    curr_step:str
    agent_response:str
    documents:list[str]