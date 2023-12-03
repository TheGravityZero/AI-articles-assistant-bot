import os
import pinecone # Vector database
import getpass
import time
import transformers

from torch import cuda, bfloat16
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA


HF_TOKEN = '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'
PINECONE_KEY = 'abc74bd6-a80a-4bdb-803c-6cf4256b5ed6'
PINECONE_ENVIRONMENT = 'gcp-starter'

model_id='meta-llama/Llama-2-13b-chat-hf'
index_name = 'llama-2-rag'
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

def load_model(model_id, HF_TOKEN):
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=HF_TOKEN
    )
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=HF_TOKEN
    )
    model.eval()
    return model

def load_tokenizer(model_id, HF_TOKEN):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=HF_TOKEN)
    return tokenizer


def init_index(index_name, PINECONE_KEY, PINECONE_ENVIRONMENT):
    pinecone.init(api_key=PINECONE_KEY, environment=PINECONE_ENVIRONMENT)
    return pinecone.Index(index_name)





def init_embedding_model(embed_model_id, device):
    return HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )

class AI_assistent():
    def __init__(self, model, tokenizer, index, embed_model):
        self.generate_response_pipeline = transformers.pipeline(
            model=model
            tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # mex number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )
        self.llm = HuggingFacePipeline(pipeline=self.generate_response_pipeline)
        self.index = index
        self.vectorstore = Pinecone(
            index, embed_model.embed_query, 'text'
        )
        
        self.rag_pipeline = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type='stuff',
            retriever=self.vectorstore.as_retriever()
        )
    
    def generate_response(self, msg, rag_flag=True):
        if rag_flag:
            return self.rag_pipeline(msg) 
        return self.llm(msg)
    
    def ckeck_index(self, index):
        return self.index.describe_index_stats()
    
    
    