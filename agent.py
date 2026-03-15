from langchain_openai import ChatOpenAI


class InsightAgent:

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        self.prompt_template = """
You are an AI assistant.

Rules:
- Use ONLY the provided context.
- Answer the question directly using facts from the context.
- Combine information from multiple context chunks if needed.
- If the answer is not explicitly supported by the context, say exactly: I don't know
- Do not guess.
- Do not use outside knowledge.
- Keep the answer short and factual.
- End with: Source: <url>

Context:
{context}

Question:
{question}

Answer:
"""

    def generate(self, question, context):
        if not context or not context.strip():
            return "I don't know"

        prompt = self.prompt_template.format(
            context=context,
            question=question
        )

        response = self.llm.invoke(prompt)
        return response.content


class AgenticReasoner:

    def __init__(self, rag, agent):
        self.rag = rag
        self.agent = agent

    def answer(self, question):
        context, sources = self.rag.retrieve(question)

        # extra check: if nothing useful was retrieved
        if not context or not sources:
            answer = "I don't know"
        else:
            answer = self.agent.generate(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }