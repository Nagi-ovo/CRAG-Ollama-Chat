# CRAG Ollama Chat

<div align="center">
<img src=https://github.com/Nagi-ovo/langgraph-crag-demo/assets/101612750/ac87701d-b9e4-4c2d-9ea9-d272766069bd
width="50%">
</div>

> create by ideogram.ai

## Preview

<div align="center">
  <video src="https://github.com/Nagi-ovo/CRAG-Ollama-Chat/assets/101612750/d0f39bdb-00b3-4626-9087-b85f4930cc53" controls>
    Your browser does not support the video tag.
  </video>
</div>


Run the demo by :

1. Creat a `config.yaml` file with the format of `config.example.yaml` and fill in the required config:

```yaml
# APIs
openai_api_key: "sk-"
openai_api_base: "https://api.openai.com/v1/chat/completions" # Or your own proxy
google_api_key: "your_google_api_key" # Unnecessary
tavily_api_key: "tvly-" # Which you can create on https://app.tavily.com/

# Ollama Config
run_local: "Yes" # Yes or No, must have ollama in ur PC
local_llm: "openhermes" # mistral, llama2 ...

# Model Config
models: "openai" # If you want to achieve the best results

# Document Config
# Support multiple websites reading
doc_url: 
  - "https://nagi.fun/llm-5-transformer"  
  - "https://nagi.fun/llm-4-wavenet"  
```

2. run the command below:

```zsh
streamlit run app.py
```

## References

- [langchain-ai/langgraph_crag_mistral](https://github.com/langchain-ai/langgraph/blob/2b42407f055dbb77331de46fe3a632ea24551347/examples/rag/langgraph_crag_mistral.ipynb)
