from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import gradio as gr
from gradio_pdf import PDF
import os

api_key = os.getenv('MISTRAL_API_KEY')
llm = MistralAI(api_key=api_key, model="mistral-large-latest")
embed_model = MistralAIEmbedding(model_name='mistral-embed', api_key=api_key)

Settings.llm = llm
Settings.embed_model = embed_model

def qa(question: str, doc: str) -> str:
    my_pdf = SimpleDirectoryReader(input_files=[doc]).load_data()
    my_pdf_index = VectorStoreIndex.from_documents(my_pdf)
    my_pdf_engine = my_pdf_index.as_query_engine()
    response = my_pdf_engine.query(question)
    return response

demo = gr.Interface(
    qa,
    [gr.Textbox(label="Question"), PDF(label="Document")],
    gr.Textbox())

if __name__ == "__main__":
    demo.launch()
