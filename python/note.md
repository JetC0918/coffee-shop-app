## Running the model locally
### Via Ollama - Lightweight
1. Install Ollama 
`bash curl -fsSL https://ollama.com/install.sh | sh`

2. Pull the LLaMA model
`bash ollama pull llama3:8b`

3. Start the Model
`bash ollama run llama3:8b`

4. Start the openai wrapper
`bash uvicorn ollama_openai_wrapper:app --host 0.0.0.0 --port 8080`

### Via vLLM  - OpenAI-compatible API server
### !!! Python version must be lower than 3.12
1. Install vllm
`bash pip intsll vllm`

2. Run vllm
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B-Instruct
```

#### Optional
#### Using Docker
```bash
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token) \
  vllm/vllm-openai \
  --model meta-llama/Llama-3.2-1B-Instruct
```