import ollama

question = {
    "role": "user",
    "content": "does the attacking player go first in fights if no charges have happened and there is still ongoing combat?",
}
llamaChat = ollama.chat(model="qwen2.5:7b", messages=[question])

print(llamaChat["message"]["content"])

embedLlama = ollama.embed(
    model="nomic-embed-text", input="This will be a Warhammer 40k RAG."
)
vector = embedLlama["embeddings"][0]

print(len(vector))
print(vector[0:5])
