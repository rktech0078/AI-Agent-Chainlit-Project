import chainlit as cl
import os 
from dotenv import load_dotenv, find_dotenv
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, Agent, Runner
from agents.run import RunConfig
from openai.types.responses import ResponseTextDeltaEvent

# Load environment
load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

# if not gemini_api_key:
#     raise ValueError("ERROR")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

main_agent = Agent(
    name="AI ASSISTANT",
    instructions="You are a helpful assistant",
    model=model,
    tools=[]
)



@cl.on_chat_start
async def chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am an agent from Abdul Rafay Shop, how can I help you today?  ").send()


@cl.on_message
async def main(message: cl.Message):
    
    history = cl.user_session.get("history")
    
    msg = cl.Message(content="")
    await msg.send()
    
    history.append({
        "role": "user",
        "content": message.content
    })
    
    result = Runner.run_streamed(
        input=history,
        starting_agent=main_agent,
        run_config=config,
    )
    
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
        
        
    history.append({
        "role": "assistant",
        "content": result.final_output
    })
    cl.user_session.set("history", history)
    
    # await cl.Message(content=result.final_output).send()