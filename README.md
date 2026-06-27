# DermAI Backend

> AI-powered dermatology analysis API — built with FastAPI, free open-source models, and no external paid APIs.

## Features

| Feature | Details |
|---------|---------|
| 🔍 **Image Analysis** | ViT (Vision Transformer) model classifies skin conditions from photos |
| 🩺 **Symptom Checker** | Rule-based engine with 11 curated skin conditions and weighted scoring |
| 🔐 **JWT Auth** | Stateless authentication with bcrypt passwords, no sessions |
| 📊 **Patient History** | Store and retrieve past analyses per user |
| 📄 **PDF Reports** | Downloadable branded reports with ICD-10 codes and recommendations |
| ⚡ **Async** | Fully async FastAPI + aiosqlite for high concurrency |
| 📖 **Auto Docs** | Swagger UI at `/docs`, ReDoc at `/redoc` |

## Quick Start

### 1. Clone & Set Up Environment

```bash
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example config
copy .env.example .env   # Windows
cp .env.example .env     # Linux/macOS

# Edit .env and set your SECRET_KEY
# Generate a strong key with:
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> **Note**: On first run, the ViT model (~330 MB) will be downloaded from HuggingFace and cached in `./model_cache/`. Subsequent starts are instant.

The API will be available at:
- **API Base**: `http://localhost:8000/api/v1`
- **Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

---

## API Endpoints

### Authentication

| Method | Endpoint | Auth | Description |
|--------|---------|------|-------------|
| `POST` | `/api/v1/auth/register` | ❌ | Register a new account |
| `POST` | `/api/v1/auth/login` | ❌ | Login, receive JWT token |
| `GET` | `/api/v1/auth/me` | ✅ | Get current user profile |

### Analysis

| Method | Endpoint | Auth | Description |
|--------|---------|------|-------------|
| `POST` | `/api/v1/analysis/image` | ✅ | Upload skin image for AI analysis |
| `POST` | `/api/v1/analysis/symptoms` | ✅ | Submit symptoms for analysis |
| `GET` | `/api/v1/analysis/symptoms/known` | ❌ | List all recognised symptom keywords |

### History

| Method | Endpoint | Auth | Description |
|--------|---------|------|-------------|
| `GET` | `/api/v1/history` | ✅ | List paginated analysis history |
| `GET` | `/api/v1/history/{id}` | ✅ | Get a single analysis |
| `DELETE` | `/api/v1/history/{id}` | ✅ | Delete an analysis |

### Reports

| Method | Endpoint | Auth | Description |
|--------|---------|------|-------------|
| `GET` | `/api/v1/reports/{id}/pdf` | ✅ | Download PDF report |

---

## Example Usage

### Register & Login

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"full_name": "Jane Doe", "email": "jane@example.com", "password": "Str0ngPass123"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "jane@example.com", "password": "Str0ngPass123"}'
```

### Symptom Analysis

```bash
curl -X POST http://localhost:8000/api/v1/analysis/symptoms \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["itching", "redness", "scaling"],
    "body_area": "forearm",
    "duration_days": 14
  }'
```

### Image Analysis

```bash
curl -X POST http://localhost:8000/api/v1/analysis/image \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@/path/to/skin_photo.jpg"
```

### Download PDF Report

```bash
curl -O -J http://localhost:8000/api/v1/reports/1/pdf \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

---

## Project Structure

```
dermai-backend/
├── app/
│   ├── main.py                  # FastAPI app + lifespan startup
│   ├── core/
│   │   ├── config.py            # Settings (pydantic-settings)
│   │   ├── database.py          # Async SQLAlchemy engine
│   │   └── security.py          # JWT + bcrypt utilities
│   ├── models/                  # SQLAlchemy ORM models
│   │   ├── user.py
│   │   └── analysis.py
│   ├── schemas/                 # Pydantic v2 request/response schemas
│   │   ├── user.py
│   │   └── analysis.py
│   ├── routers/                 # FastAPI route handlers
│   │   ├── auth.py
│   │   ├── analysis.py
│   │   ├── history.py
│   │   └── reports.py
│   ├── services/                # Business logic layer
│   │   ├── auth_service.py
│   │   ├── ml_service.py
│   │   ├── symptom_service.py
│   │   ├── analysis_service.py
│   │   └── report_service.py
│   └── ml/
│       └── model_loader.py      # Singleton ViT model loader
├── requirements.txt
├── .env.example
└── README.md
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | *(required)* | JWT signing key — **change in production!** |
| `ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | JWT TTL |
| `DATABASE_URL` | `sqlite+aiosqlite:///./dermai.db` | Database connection URL |
| `MODEL_NAME` | `google/vit-base-patch16-224` | HuggingFace model name |
| `HF_CACHE_DIR` | `./model_cache` | Local model cache directory |
| `UPLOAD_DIR` | `./uploads` | Directory for uploaded images |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum image upload size |
| `ALLOWED_ORIGINS` | `http://localhost:3000,...` | CORS allowed origins (comma-separated) |

---

## Technology Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** — Async web framework with auto-docs
- **[SQLAlchemy 2.x](https://www.sqlalchemy.org/)** — Async ORM with aiosqlite
- **[HuggingFace Transformers](https://huggingface.co/google/vit-base-patch16-224)** — ViT image classification model
- **[python-jose](https://python-jose.readthedocs.io/)** — JWT token handling
- **[passlib](https://passlib.readthedocs.io/)** — bcrypt password hashing
- **[ReportLab](https://www.reportlab.com/)** — PDF generation
- **[Pydantic v2](https://docs.pydantic.dev/)** — Data validation

## License

MIT
