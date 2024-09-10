import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# Embedding class
class HFEmbeddings:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts):
        return self.model.encode(texts)

# Language model class
class HFChatModel:
    def __init__(self, model_name='distilgpt2', temperature=0.7):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, config={'temperature': temperature})
    
    def generate(self, prompt, max_length=150):
        return self.generator(prompt, max_length=max_length)[0]['generated_text']

# Custom LLM Chain
class CustomLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, **kwargs):
        prompt_text = self.prompt.format(**kwargs)
        return self.llm.generate(prompt_text)

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="sales_response.csv")
documents = loader.load()

# Replace OpenAIEmbeddings with HFEmbeddings
embeddings = HFEmbeddings()
db = FAISS.from_documents(documents, embeddings.embed)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 3. Setup LLMChain & prompts
# Replace ChatOpenAI with HFChatModel
llm = HFChatModel()

template = """
You are a design creation assistant for AnDAPT. Your task is to map user inputs into an
optimal power solution using specified components while adhering to design rules and
constraints.

1/ Response should be very similar or even identical to the past best responses, 
in terms of output, previous rules, and other patterns that occur throughout the dataset

2/ If the best responses are irrelevant, then try to recognize the pattern that the data has and give an output acoordingly

Below is a message I received from the prospect:
{message}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{best_practice}

Please write the best practice that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

# Use the new LLM class
chain = CustomLLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Customer response generator", page_icon=":bird:")

    st.header("Customer response generator :bird:")
    message = st.text_area("customer message")

    if message:
        st.write("Generating best practice message...")
        result = generate_response(message)
        st.info(result)

if __name__ == '__main__':
    main()
