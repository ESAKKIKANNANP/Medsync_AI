#!/usr/bin/env python3
"""
MedSync AI - System Diagnostics
Check connectivity and health of all system components
"""

import sys
import time
from pathlib import Path

# Add root to path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

import requests

print("=" * 60)
print("🔍 MedSync AI System Diagnostics")
print("=" * 60)
print()

# Test 1: Check Ollama
print("1️⃣ Testing Ollama (LLM Service)...")
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=10)
    if response.status_code == 200:
        data = response.json()
        models = data.get("models", [])
        print("   ✅ Ollama is ONLINE")
        print(f"   📦 Available models: {len(models)}")
        for model in models:
            model_name = model.get("name", "unknown")
            print(f"      - {model_name}")
    else:
        print(f"   ❌ Ollama returned status {response.status_code}")
except requests.exceptions.Timeout:
    print("   ⚠️ Ollama health check TIMEOUT")
    print("      Ollama may be starting or busy with inference")
except requests.exceptions.ConnectionError:
    print("   ❌ Cannot connect to Ollama at http://localhost:11434")
    print("      Make sure Ollama is running: ollama serve")
except Exception as e:
    print(f"   ❌ Ollama check failed: {e}")

print()

# Test 2: Check Backend API
print("2️⃣ Testing Backend API...")
try:
    response = requests.get("http://localhost:8000/health", timeout=15)
    if response.status_code == 200:
        data = response.json()
        llm_ok = data.get("llm_available", False)
        print("   ✅ Backend API is ONLINE")
        print(f"   {'✅' if llm_ok else '❌'} LLM availability: {'ONLINE' if llm_ok else 'OFFLINE'}")
        print(f"   📊 Vector DB stats: {data.get('vector_db_collections', {})}")
    else:
        print(f"   ⚠️ Backend returned status {response.status_code}")
except requests.exceptions.Timeout:
    print("   ⚠️ Backend API TIMEOUT")
    print("      Make sure backend is running: python -m medsync_ai.api.server")
except requests.exceptions.ConnectionError:
    print("   ❌ Cannot connect to Backend at http://localhost:8000")
    print("      Make sure backend is running: ./run_backend.sh")
except Exception as e:
    print(f"   ❌ Backend check failed: {e}")

print()

# Test 3: Check Frontend (Streamlit)
print("3️⃣ Testing Frontend (Streamlit)...")
try:
    response = requests.get("http://localhost:8502", timeout=5)
    if response.status_code == 200:
        print("   ✅ Frontend is ONLINE at http://localhost:8502")
    else:
        print(f"   ⚠️ Frontend returned status {response.status_code}")
except requests.exceptions.Timeout:
    print("   ⚠️ Frontend TIMEOUT")
except requests.exceptions.ConnectionError:
    print("   ❌ Cannot connect to Frontend at http://localhost:8502")
    print("      Make sure Streamlit is running: ./run_frontend.sh")
except Exception as e:
    print(f"   ❌ Frontend check failed: {e}")

print()
print("=" * 60)
print("📋 Diagnostic Summary:")
print("=" * 60)
print()
print("✅ All running? Open: http://localhost:8502")
print()
print("❌ Ollama offline? Run:")
print("   ollama serve")
print()
print("❌ Backend offline? Run:")
print("   ./run_backend.sh")
print()
print("❌ Frontend offline? Run:")
print("   ./run_frontend.sh")
print()
print("=" * 60)
