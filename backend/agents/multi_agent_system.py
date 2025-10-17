import asyncio
from re import S
from uagents import Agent, Context, Model, Bureau
from backend.langgraph_logic.langgraph_example import WeatherWorkFlow

#/Users/mceballos456/fetchai-playground/backend/langgraph_logic

class Question(Model):
    question:str

class Answer(Model):
    answer:str
    question:str

coordinator = Agent(name="coordinator", seed="coordinator_weather_seed")

weather_rag_agent = Agent(name = 'weather_rag_agent', seed = 'weather_rag_agent_seed')

workflow = None

@coordinator.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info("Starting up coordinator agent...")

@weather_rag_agent.on_event("startup")
async def startup(ctx:Context):
    ctx.logger.info("Starting up weather rag agent...")
    global workflow

    workflow = WeatherWorkFlow()
    ctx.logger.info("Workflow initialized")

@weather_rag_agent.on_message(model = Question)
async def on_question(ctx:Context, question:Question):
    ctx.logger.info(f"Received question: {question.question}")
    answer = workflow.query(question.question)
    ctx.logger.info(f"Answer: {answer}")
    await ctx.send(Answer(answer=answer, question=question.question))

@coordinator.on_message(model = Answer)
async def on_answer(ctx:Context, answer:Answer):
    ctx.logger.info(f"Received answer: {answer.answer}")
    await ctx.send(Answer(answer=answer.answer, question=answer.question))

@coordinator.on_event("shutdown")
async def shutdown(ctx:Context):
    ctx.logger.info("Shutting down coordinator agent...")
    global workflow
    if workflow:
        workflow.shutdown()
    ctx.logger.info("Workflow shutdown")

if __name__ == "__main__":
    bureau = Bureau()
    bureau.add(coordinator)
    bureau.add(weather_rag_agent)
    bureau.run()
    print("Bureau running...")
