#!/bin/bash
# Start MedSync AI Frontend with Streamlit

set -e  # Exit on error

cd "$(dirname "$0")/frontend"

# Activate virtual environment
source ../venv/bin/activate

# Clear any lingering Streamlit processes
echo "🧹 Cleaning up old Streamlit processes..."
pkill -f streamlit || true
sleep 1

echo ""
echo "=========================================="
echo "🚀 Starting MedSync AI Surgical Consultant"
echo "=========================================="
echo ""
echo "📍 Frontend running on: http://localhost:8502"
echo ""
echo "⚠️  IMPORTANT: Make sure these are running:"
echo "   1. Backend: ./run_backend.sh (port 8000)"
echo "   2. Frontend: This script (port 8502)"
echo "   3. Set GEMINI_API_KEY environment variable"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Check if backend is running
echo "🔍 Checking backend connectivity..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
  echo "✅ Backend is reachable at localhost:8000"
  # Wait a moment and check Gemini API status
  sleep 1
  if curl -s http://localhost:8000/health | grep -q '"gemini_api_available": true'; then
    echo "✅ Gemini API is available"
  else
    echo "⚠️  Backend is running but Gemini API shows unavailable"
    echo "   Make sure GEMINI_API_KEY is set correctly"
  fi
else
  echo "❌ WARNING: Cannot reach backend at localhost:8000"
  echo "   Start the backend first: ./run_backend.sh"
  echo "   Frontend will still work but cannot reach LLM"
  sleep 3
fi

echo ""
echo "Starting Streamlit frontend..."
echo ""

# Run Streamlit on port 8502
streamlit run main.py
