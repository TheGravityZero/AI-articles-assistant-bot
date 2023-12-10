import os
import pinecone  # Vector database
import getpass
import time
import transformers

from torch import cuda, bfloat16
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

from settings import MODEL_INIT_SETTINGS, HF_TOKEN, PINECONE_TOKEN, logger


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class AI_Assistant(metaclass=Singleton):
    def __init__(self, settings=MODEL_INIT_SETTINGS):
        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        logger.info(f"Device: {self.device}")

        self.generate_response_pipeline = transformers.pipeline(
            model=self._load_model(settings['model_id'], HF_TOKEN),
            tokenizer=self._load_tokenizer(settings['model_id'], HF_TOKEN),
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            temperature=0.01,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # mex number of tokens to generate in the output
            repetition_penalty=1.1,  # without this output begins repeating
        )

        self.llm = HuggingFacePipeline(pipeline=self.generate_response_pipeline)
        self.index = self._init_index(settings['index_id'], settings['pinecone_environment_id'], PINECONE_TOKEN)
        self.embed_model = self._init_embedding_model(settings['embed_model_id'])
        self.vectorstore = Pinecone(
            self.index, self.embed_model.embed_query, 'text',
        )
        
        self.rag_pipeline = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type='stuff',
            retriever=self.vectorstore.as_retriever(),
        )
    
    def generate_response(self, msg, rag_flag=True):
        if rag_flag:
            return self.rag_pipeline(msg).get('result')
        return self.llm(msg)
    
    def check_index(self):
        return self.index.describe_index_stats()

    def _load_model(self, model_id, hf_token=HF_TOKEN):
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        )

        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            use_auth_token=hf_token,
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_token,
        )
        model.eval()
        return model

    def _load_tokenizer(self, model_id, hf_token=HF_TOKEN):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            use_auth_token=hf_token,
        )
        return tokenizer

    def _init_index(self, index_name, pinecone_environment, pinecone_key=PINECONE_TOKEN):
        pinecone.init(api_key=pinecone_key, environment=pinecone_environment)
        return pinecone.Index(index_name)

    def _init_embedding_model(self, embed_model_id):
        return HuggingFaceEmbeddings(
            model_name=embed_model_id,
            model_kwargs={'device': self.device},
            encode_kwargs={'device': self.device, 'batch_size': 32}
        )
