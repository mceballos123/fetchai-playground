from datetime import datetime
from uagents import Agent, Context, Protocol
from dotenv import load_dotenv
import os
from uuid import uuid4
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

# from ollama import Ollama
from backend.langgraph_logic.langgraph_example import WeatherWorkFlow


load_dotenv()

Weather_Seed = os.getenv("WEATHER_SEED_PHRASE")
Weather_Endpoint = os.getenv("WEATHER_ENDPOINT")

weather_agent = Agent(name="weather_agent", seed=Weather_Seed, port=8001, mailbox=True)
# , endpoint=Weather_Endpoint

weather_protocol = Protocol(name="weather_protocol", spec=chat_protocol_spec)

workflow = None


# on message that takes in the chat message from the user
@weather_protocol.on_message(
    ChatMessage
)  # Chat messages is messages from the user to the agent
async def on_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    global workflow
    print(f"Received chat message from {sender}: {msg}")
    print(f"Received chat message from {sender}: {msg.content}")

    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )

    ctx.logger.info(f"Received chat message from %s: %s", sender, msg)
    text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
    if not text_parts:
        await ctx.send(
            sender,
            ChatMessage(
                timestamp=datetime.now(),
                msg_id=str(uuid4()),
                content=[TextContent(type="text", text="Please send a text message.")],
            ),
        )
        return

    question = text_parts[0]
    ctx.logger.info(f"Received chat question: {question}")
   
    if workflow is None:
        workflow = WeatherWorkFlow()
        print(f"Workflow initialized")

    answer = workflow.query(question)
    
    ctx.logger.info(f"Sending answer: {answer}")

    # We send the answer to the user and end the session

    await ctx.send(
        sender,
        ChatMessage(
            timestamp=datetime.now(),
            msg_id=str(uuid4()),
            content=[
                TextContent(type="text", text=answer),
                EndSessionContent(type="end-session"),
            ],
        ),
    )


# Handle acknowledgements as required by the chat protocol
# Handles the acknowledgement from the user to the agent
@weather_protocol.on_message(ChatAcknowledgement)
async def on_ack(ctx: Context, sender: str, ack: ChatAcknowledgement):
    ctx.logger.info("Received ChatAcknowledgement from %s", sender)


weather_agent.include(weather_protocol, publish_manifest=True)

if __name__ == "__main__":
    weather_agent.run()
