# nbscribe - Testing Procedures

## Environment Setup

### First Time Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate nbscribe
```

### Update Existing Environment
```bash
# Update environment with new dependencies
conda env update -f environment.yml --prune
conda activate nbscribe
```

## Basic Testing

### 1. Start Server
```bash
python main.py
```
Expected: Server starts on port 5317 without errors

### 2. Health Check
```bash
curl http://localhost:5317/health
```
Expected: `{"status":"healthy","service":"nbscribe","version":"0.1.0"}`

### 3. API Info
```bash
curl http://localhost:5317/api/info
```
Expected: JSON with endpoints list

### 4. Chat API Test
```bash
curl -X POST http://localhost:5317/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, test!"}'
```
Expected: Real AI response (not echo)

### 5. Web Interface
```bash
open http://localhost:5317
```
Expected: Chat interface loads, can send messages and receive AI responses

## Prerequisites
- `OPENAI_API_KEY` environment variable set
- Python 3.12+
- Conda or pip

## Quick Verification
1. Environment updates successfully
2. Server starts without errors  
3. Health check returns healthy status
4. Chat interface works with real AI responses

## Generate Frozen Requirements
Once everything is working, freeze the versions:
```bash
pip freeze > requirements-frozen.txt
``` 