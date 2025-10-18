from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from backend.rag.rag_system import WeatherRAGSystem



# AgentState keeps track of the state of the workflow for the graph
class AgentState(TypedDict):
    messages:Annotated["list", add_messages]

    question:str #gets the questions from the user
    answer:str # gets the answer from the RAG SYSTEM
    context:str # gets the context from the RAG SYSTEM


# WeatherWorkFlow is the main class that initializes the workflow and adds the nodes and edges
class WeatherWorkFlow:

    def __init__(self):
        print("Initializing Weather Workflow...")

        self.rag_system = WeatherRAGSystem() #intilaizes the RAG SYSTEM

        self.workflow = StateGraph(AgentState) # Keeps track of the state of the workflow for the graph
    # State keeps track of the state of the workflow for the graph, State keeps the previous and current state of the workflow
        self.workflow.add_node("extract_question", self.extract_question)
        self.workflow.add_node("search_documents", self.search_documents)
        self.workflow.add_node("generate_answer", self.generate_answer)

        

        # Define the flow
        self.workflow.add_edge(START, "extract_question")
        self.workflow.add_edge("extract_question", "search_documents")
        self.workflow.add_edge("search_documents", "generate_answer")
        self.workflow.add_edge("generate_answer", END)
        
        # Compile the graph
        self.app = self.workflow.compile()
        print("✅ Workflow Ready!")

    # 
    
    def extract_question(self, state: AgentState) -> AgentState:
        """Extract the question from messages"""
        print("\n📝 Step 1: Extracting question...")
        last_message = state["messages"][-1]
        question = last_message.content if hasattr(last_message, 'content') else str(last_message)
        print(f"   Question: {question}")
        return {"question": question}
    
    def search_documents(self, state: AgentState) -> AgentState:
        """Search documents using RAG"""
        print("🔍 Step 2: Searching documents...")
        question = state["question"]
        context = self.rag_system.query(question)
        print(f"   Found context (length: {len(context)} chars)")
        return {"context": context}
    
    def generate_answer(self, state: AgentState) -> AgentState:
        """Generate final answer"""
        print("💡 Step 3: Generating answer...")
        answer = state["context"]  # In this simple version, RAG answer IS our final answer
        
        # Add to messages
        ai_message = AIMessage(content=answer)
        print(f"   Answer ready!")
        return {
            "answer": answer,
            "messages": [ai_message]
        }
    
    def query(self, question: str) -> str:
        """Run the workflow"""
        print("\n" + "="*80)
        print(f"🏀 Processing: {question}")
        print("="*80)
        
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "question": "",
            "answer": "",
            "context": ""
        }
        
        result = self.app.invoke(initial_state)
        return result["answer"]


if __name__ == "__main__":
    workflow = WeatherWorkFlow()

    question = "What is the weather like in San Jose, CA? for this month of October 2025?"
    answer = workflow.query(question)
    print(f"Answer:{answer}")