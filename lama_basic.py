from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer
# use Huggingface embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from huggingface_hub import hf_hub_download


# import
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb


# model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
# model_url = "https://huggingface.co/Undi95/Mistral-ClaudeLimaRP-v3-7B-GGUF/resolve/main/Mistral-ClaudeLimaRP-v3-7B.q4_k_s.gguf"
model_url = "/mnt/d/Data/Models/Mistral-ClaudeLimaRP-v3-7B.q4_k_s.gguf",

model_path = hf_hub_download(
    repo_id="TheBloke/WizardLM-7B-uncensored-GGUF", filename="WizardLM-7B-uncensored.Q4_0.gguf", local_dir="/mnt/d/Data/Models/"
)

prompt = """

"""
# tokenizer_model_path = hf_hub_download(
#     repo_id="mistralai/Mistral-7B-v0.1", filename="Mistral-7B-v0.1", local_dir="/mnt/d/Data/Models/"
# )
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

set_global_tokenizer(tokenizer)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=model_path,
    temperature=0.7,
    max_new_tokens=2048,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# load documents
# chroma_client = chromadb.EphemeralClient()
# chroma_collection = chroma_client.create_collection("quickstart")


# db = chromadb.PersistentClient(path="./chroma_db")
# chroma_collection = db.get_or_create_collection("quickstart")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# # Save documents
# documents = SimpleDirectoryReader("/mnt/d/Data/athena").load_data()
# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context, embed_model=embed_model
# )
# index = VectorStoreIndex.from_vector_store(
#     vector_store,
#     embed_model=embed_model,
# )

# set up query engine
# query_engine = index.as_query_engine(llm=llm)

# response = query_engine.query("relation between percy and athena ?")
# print(response)

response_iter = llm.stream_complete(prompt)
for response in response_iter:
    print(response.delta, end="", flush=True)
