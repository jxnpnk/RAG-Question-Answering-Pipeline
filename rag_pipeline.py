import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class RAGPipeline:
    def __init__(self, data_path):
        self.embeddings = OpenAIEmbeddings()

        # Load dataset
        self.df = pd.read_csv(data_path)

        print("\nLoaded file:", data_path)
        print("\nColumns:")
        print(self.df.columns.tolist())
        print("\nNumber of rows:", len(self.df))
        print("\nFirst 5 rows:")
        print(self.df.head())

        print("\nRows containing hiking or surfing:")
        print(
            self.df[
                self.df["text"].astype(str).str.contains(
                    "hiking|surfing", case=False, na=False
                )
            ]
        )

        print("\nRows containing exercise:")
        print(
            self.df[
                self.df["text"].astype(str).str.contains(
                    "exercise", case=False, na=False
                )
            ]
        )

        documents = []
        for _, row in self.df.iterrows():
            doc = Document(
                page_content=str(row["text"]),
                metadata={
                    "source": row["source_url"],
                    "index": row["index"],
                },
            )
            documents.append(doc)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )

        split_docs = splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)

    def retrieve(self, query, k=20):
        results = self.vectorstore.similarity_search(query, k=k)

        print("\nRetrieved documents:")
        for i, doc in enumerate(results, start=1):
            print(f"\n--- Result {i} ---")
            print("Source:", doc.metadata.get("source"))
            print("Text:", doc.page_content[:500])

        spain_docs = []
        for doc in results:
            source = doc.metadata.get("source", "")
            if "so-into-northern-spain" in source:
                spain_docs.append(doc)

        if spain_docs:
            results = spain_docs[:5]

        context = "\n\n".join(
            [
                f"Source: {doc.metadata['source']}\nContent: {doc.page_content}"
                for doc in results
            ]
        )
        sources = [doc.metadata["source"] for doc in results]

        return context, sources