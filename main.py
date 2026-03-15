from rag_pipeline import RAGPipeline
from agent import InsightAgent, AgenticReasoner


rag = RAGPipeline("/Users/jeanngugi/Desktop/Uni/Masters/Personal_projects/RAG_dataset/documents.csv")

agent = InsightAgent()
reasoner = AgenticReasoner(rag, agent)

question = "What are the risk classifications for AI?"

result = reasoner.answer(question)

print("\nANSWER")
print(result["answer"])

print("\nSOURCES")
print(result["sources"])

