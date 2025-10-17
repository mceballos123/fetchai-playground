import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


class WeatherRAGSystem:
    def __init__(
        self, documents_path="/Users/mceballos456/fetchai-playground/backend/documents"
    ):
        print(f"Initializing Weather RAG System with documents at {documents_path}")

        self.llm = Ollama(model="llama3.2:1b", request_timeout=120)
        print(f"LLM:{self.llm}")
        # represents your documents in a vector space using a number, takes text as input
        # conver text to words and if a word is similar to another, it will be reprensed
        # converts question into numbers
        # finds documents with similar numbers and returns the documents
        self.embed_model = OllamaEmbedding(model_name="llama3.2:1b")
        print(f"Embed Model:{self.embed_model}")
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512  # word count

        Settings.chunk_overlap = 50

        # printing the documents path
        print(f"Loading documents from {documents_path}")
        # loads the documents from the documents folder
        documents = SimpleDirectoryReader(documents_path).load_data()

        # create vector index and stores in memory
        self.index = VectorStoreIndex.from_documents(documents)
        print(f"Index:{self.index}")
        # query engine is used to query the index

        self.query_engine = self.index.as_query_engine()
        print(f"Query Engine:{self.query_engine}")

    def query(self, question: str) -> str:

        print(f"Question:{question}")
        response = self.query_engine.query(question)

        print(f"Response:{response}")
        return str(response)


if __name__ == "__main__":
    rag_system = WeatherRAGSystem()

    print("Testing RAG System...")

    test_questions = "What is the weather like in San Jose, CA?"

    answer = rag_system.query(test_questions)
    if not answer:
        print("Error: No answer found")
    print(answer)
    print(f"Answer:{answer}")
