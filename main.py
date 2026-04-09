from data_loader import load_file
from chunking import DocumentChunker
from rag_chatbot import RAGChatBot

if __name__ == "__main__":
    bot = RAGChatBot()

    # Step 1: Load file
    file_path = input("📂 Enter file path: ")
    docs = load_file(file_path)

    # Step 2: Chunk
    chunker = DocumentChunker()
    chunks = chunker.split_documents(docs)

    # Step 3: Load into vector store
    bot.load_documents(chunks)

    print("\n💬 Chat started! (type 'exit' to quit)\n")

    # Step 4: Chat loop
    while True:
        question = input("You: ")

        if question.lower() == "exit":
            break

        if question.lower() == "upload":
            file_path = input("📂 Enter new file path: ")
            docs = load_file(file_path)
            chunks = chunker.split_documents(docs)
            bot.load_documents(chunks)
            print("✅ New document loaded!\n")
            continue

        result = bot.ask(question)
        print(f"\n🤖 Answer: {result['answer']}")
        print(f"📄 Sources: {result['sources']}\n")