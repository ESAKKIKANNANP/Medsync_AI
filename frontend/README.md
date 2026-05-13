# MedSync AI Frontend - Streamlit Dashboard

A comprehensive Streamlit-based web interface for the MedSync AI surgical intelligence platform. This frontend provides a professional, high-tech medical dashboard with dark mode theming and real-time integration with the FastAPI backend.

## Features

### 🏥 Core Components

1. **System Dashboard**
   - Real-time system health status
   - LLM engine availability monitoring
   - Vector database collection metrics
   - Backend connectivity validation

2. **Patient Registration & Management**
   - Comprehensive patient intake forms
   - Risk factor assessment
   - Medical history tracking
   - Real-time patient data synchronization

3. **Unified Clinical View**
   - Patient demographics display
   - Lab results visualization
   - Medication and comorbidity summary
   - Risk factor calculation

4. **Surgical Roadmap Timeline**
   - Visual surgical phase progression
   - Real-time risk score calculation
   - Dynamic phase navigation
   - Hemorrhage risk assessment based on patient factors

5. **Surgical Vision Analysis (VQLA)**
   - Real-time surgical frame analysis
   - Instrument detection and tracking
   - Confidence score visualization
   - Scene understanding integration

6. **AI Surgical Assistant**
   - Multi-turn chat interface
   - Query-based decision support
   - Reasoning path visualization
   - Context-aware responses based on patient data

7. **Active Alerts System**
   - Priority-based alert generation
   - Risk factor recommendations
   - Clinical decision support messaging

### 🎨 UI/UX Features

- **Dark Mode Theme**: Professional high-tech medical aesthetic
- **Custom CSS**: Gradient buttons, glowing borders, smooth transitions
- **Responsive Layout**: Optimized for various screen sizes
- **Accessible Design**: High contrast, clear typography
- **Real-time Updates**: Session state management for persistent data

## Installation

### Local Development

1. **Clone or navigate to the project directory:**
```bash
cd "/Users/dhanavijayan/Documents/MedSync copy"
```

2. **Create a Python virtual environment:**
```bash
python3 -m venv frontend_env
source frontend_env/bin/activate  # On Windows: frontend_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r frontend/requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env  # Create from template
# Edit .env with your API_BASE_URL and other settings
```

5. **Run the Streamlit app:**
```bash
streamlit run frontend/main.py
```

The application will be available at `http://localhost:8501`

### Docker Setup

1. **Build the Docker image:**
```bash
docker build -f frontend/Dockerfile -t medsync-frontend:latest ./frontend
```

2. **Run with Docker Compose (recommended):**
```bash
docker-compose up -d
```

This starts both the backend (port 8000) and frontend (port 8501).

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
API_BASE_URL=http://localhost:8000

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true

# Data paths
SURGICAL_IMAGES_PATH=/Users/dhanavijayan/Documents/MedSync\ copy/data/endovis2018/val/image
```

### Streamlit Config

The `.streamlit/config.toml` file controls:
- Color theme (primary, background, text)
- Server settings (port, address, SSL)
- Client behavior (logger levels, XSRF protection)
- Browser settings (usage stats, theme)

## Usage

### 1. System Dashboard

Start here to check system health and connectivity:
- Verify backend is running
- Check LLM engine status
- View vector database statistics

### 2. Patient Registration

Register a new patient with:
- Basic demographics (ID, age)
- Medication history (anticoagulation, aspirin)
- Lab results (platelet count)
- Medical history (coagulopathy, renal/hepatic impairment)

### 3. Clinical View

View comprehensive patient profile:
- Patient demographics
- Risk factor summary
- Medication details
- Organ function status

### 4. Surgical Roadmap

Track procedure progress:
- 6-phase surgical timeline
- Real-time risk scoring (0-100%)
- Phase navigation (previous/next)
- Hemorrhage risk assessment

**Risk Score Calculation:**
```
Base Risk = 0.3 (30%)
Anticoagulation: +20%
Coagulopathy: +15%
Family History: +10%
Low Platelets (<100k): +15%
Maximum Cap: 95%
```

### 5. Surgical Vision

Analyze surgical frames:
1. Select a frame from the surgical image directory
2. Click "Analyze Frame" to detect instruments
3. View confidence scores for detected instruments
4. Review scene understanding analysis

**Detected Instruments:**
- Grasper
- Bipolar
- Scissors
- Hook

### 6. AI Assistant

Interactive chat with surgical decision support:
1. Ask clinical questions (e.g., "What are the risks of hemorrhaging?")
2. System retrieves relevant medical knowledge from vector store
3. LLM generates context-aware responses
4. View reasoning path for transparency

**Example Queries:**
- "What are the bleeding risks for this patient?"
- "Optimal hemostasis technique for this case?"
- "Pre-operative optimization recommendations?"

### 7. Alerts

View active clinical alerts:
- Anticoagulation status
- Coagulopathy warnings
- Low platelet alerts
- Family history flags

Color-coded by severity:
- 🔴 **HIGH**: Immediate action required (red)
- 🟡 **MEDIUM**: Enhanced monitoring (orange)
- 🟢 **LOW**: Standard precautions (green)

## API Integration

### Backend Endpoints Used

```
GET  /health                      - System health check
POST /api/patient/register        - Register new patient
POST /api/surgeon/query           - Query surgical assistant
POST /api/surgical-vision/analyze - Analyze surgical frame
```

### Request/Response Examples

**Patient Registration:**
```json
POST /api/patient/register
{
  "patient_id": "p001",
  "age": 58,
  "on_anticoagulation": true,
  "on_aspirin": false,
  "platelet_count": 95000,
  "has_coagulopathy": false,
  "mh_family_history": false,
  "has_renal_impairment": false,
  "has_hepatic_impairment": false
}
```

**Surgeon Query:**
```json
POST /api/surgeon/query
{
  "query": "What are the risks of hemorrhaging during dissection?",
  "patient_id": "p001"
}

Response:
{
  "response": "Based on patient data...",
  "reasoning_path": ["Retrieved document 1", "Retrieved document 2"],
  "confidence": 0.85
}
```

**Surgical Vision Analysis:**
```json
POST /api/surgical-vision/analyze
Files: surgical_frame.jpg

Response:
{
  "detections": [
    {
      "instrument": "grasper",
      "confidence": 0.92,
      "bbox": [100, 150, 200, 250]
    }
  ],
  "scene_understanding": {...}
}
```

## Architecture

### Component Hierarchy

```
Streamlit App (main.py)
├── Sidebar Navigation
├── Header Component
├── System Status Display
├── Patient Registration
├── Unified Clinical View
├── Surgical Roadmap
├── Surgical Vision Analysis
├── AI Chat Assistant
└── Active Alerts

API Client Layer
├── register_patient()
├── query_surgeon_Assistant()
├── analyze_surgical_frame()
└── check_system_health()

FastAPI Backend
├── Patient Management
├── RAG Engine
├── Vector Database
└── Surgical Vision Engine
```

### State Management

Uses Streamlit's `st.session_state` to maintain:
- `patient_data`: Current patient information
- `chat_history`: Conversation history
- `roadmap_data`: Surgical roadmap progress
- `current_step`: Current surgical phase
- `medical_analysis`: Latest frame analysis
- `system_status`: Backend health status

## Deployment

### Docker Compose (Production)

```bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services:
- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:8501
- **Ollama LLM**: http://localhost:11434
- **ChromaDB**: http://localhost:8003 (optional)

### Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medsync-frontend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: medsync-frontend
  template:
    metadata:
      labels:
        app: medsync-frontend
    spec:
      containers:
      - name: frontend
        image: medsync-frontend:latest
        ports:
        - containerPort: 8501
        env:
        - name: API_BASE_URL
          value: "http://medsync-backend:8000"
```

## Troubleshooting

### Issue: Backend Not Connecting

**Symptoms:** "Failed to register patient" errors

**Solutions:**
1. Verify backend is running: `curl http://localhost:8000/health`
2. Check `API_BASE_URL` environment variable
3. Restart both services: `docker-compose restart`

### Issue: Surgical Images Not Loading

**Symptoms:** "No surgical images found"

**Solutions:**
1. Verify images exist in: `data/endovis2018/val/image/`
2. Check `SURGICAL_IMAGES_PATH` in environment
3. Ensure images are `.png` or `.jpg` format

### Issue: LLM Engine Offline

**Symptoms:** "LLM Available: ⚠ Offline"

**Solutions:**
1. Start Ollama: `ollama serve`
2. Pull model: `ollama pull openhermes2.5`
3. Verify: `curl http://localhost:11434/api/tags`

### Issue: Streamlit Port Already in Use

**Solutions:**
1. Kill existing process: `lsof -ti:8501 | xargs kill -9`
2. Change port: `streamlit run main.py --server.port 8502`
3. Use Docker with different mapping: `-p 8502:8501`

## Performance Optimization

### Frontend Caching

The app uses Streamlit's caching decorators:
```python
@st.cache_resource
def load_surgical_images():
    # Cached: re-runs only when function code changes
    pass

@st.cache_data
def check_system_health():
    # Cached: re-runs periodically
    pass
```

### API Timeouts

Configure in `main.py`:
- Patient registration: 10 seconds
- Surgeon query: 30 seconds (LLM inference)
- Frame analysis: 15 seconds

### Image Processing

- Load images on-demand (not on app startup)
- Cache overlay results in session state
- Lazy-load surgical vision components

## Future Enhancements

- [ ] Real-time video stream analysis
- [ ] Multi-patient dashboard view
- [ ] Historical patient tracking
- [ ] Surgical outcome predictions
- [ ] Integration with hospital EHR systems
- [ ] Mobile-responsive design
- [ ] Advanced analytics dashboard
- [ ] Offline mode support
- [ ] Automated report generation
- [ ] Integration with surgical robotics systems

## License

MedSync AI - Multi-Modal Surgical Intelligence Framework
Licensed under the MIT License - See LICENSE.txt

## Support

For issues, questions, or feature requests:
1. Check the troubleshooting section above
2. Review logs: `cat logs/medsync.log`
3. Contact the development team

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

**Last Updated:** February 24, 2026
**Version:** 1.0.0
**Status:** Production Ready
