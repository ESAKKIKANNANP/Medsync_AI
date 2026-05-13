#!/bin/bash
# Start MedSync AI Backend API

set -e  # Exit on error

cd "$(dirname "$0")"

# Activate virtual environment  
source venv/bin/activate

echo ""
echo "=========================================="
echo "🚀 Starting MedSync AI Backend API"
echo "=========================================="
echo ""
echo "📍 Running on: http://localhost:8000"
echo "📌 Health check: http://localhost:8000/health"
echo ""
echo "⚙️  Using: Google Gemini API (Fast & Multimodal)"
echo ""
echo "⚠️  IMPORTANT: Set your Gemini API key"
echo "   export GEMINI_API_KEY=your-api-key"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
  echo "❌ WARNING: GEMINI_API_KEY environment variable not set"
  echo "   The backend will start but Gemini API will be unavailable"
  echo "   Set it with: export GEMINI_API_KEY=your-api-key"
  sleep 2
else
  echo "✅ Gemini API key detected"
fi

echo ""
echo "Starting backend..."
echo ""

# Run the backend API from project root
python -m medsync_ai.api.server
