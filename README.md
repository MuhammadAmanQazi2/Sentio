<div align="center">

 Sentio
 AI-Powered App for Early Screening of Depression using Text and Audio

[![NED University](https://img.shields.io/badge/NED%20University-2022--2026-blue?style=flat-square)](https://neduet.edu.pk)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Firebase](https://img.shields.io/badge/Firebase-Firestore-yellow?style=flat-square&logo=firebase)](https://firebase.google.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

<img width="384"  alt="image" src="https://github.com/user-attachments/assets/4d0e8ad5-fd61-4da1-9e5d-cfae2c714633" />

<img width="199"  alt="image" src="https://github.com/user-attachments/assets/916fb861-7bbe-4840-ae6f-95427feb9cdd" />

<img width="456"  alt="image" src="https://github.com/user-attachments/assets/41419e1e-23b7-4fe6-9e12-06cf9d43cb7d" />


> A non-invasive, AI-driven mental wellness companion that detects depression risk from natural conversation using multimodal deep learning.

Department of Computer Science and Information Technology 
NED University of Engineering & Technology — Batch 2022–2026

</div>




 What is Sentio?

Depression affects approximately 332 million people worldwide — yet most cases go undetected until symptoms significantly impact daily life. Sentio is an AI-powered Android application that enables early, non-invasive screening of depression risk through natural text and voice conversations.

The system analyzes what users say (text) and how they say it (audio) using a custom-trained multimodal deep learning fusion model, then calibrates an empathetic LLM-powered chatbot response to the user's detected emotional risk level. It is not a diagnostic tool — it is an awareness and support platform.


 👥 Team

| Name | Roll No | Contribution |
|---|---|---|
| Ayan Adnan | DT-22014 | Audio sub-model, MFCC extraction, ANOVA feature selection, DAIC-WOZ preprocessing |
| Syed Awais Waseem | DT-22033 | Frontend UI, dashboard visualizations, deployment |
| Muhammad Aman Qazi | DT-22044 | Text model (DistilBERT/NLP pipeline), documentation |
| Muhammad Aamir | DT-22047 | Backend (FastAPI), multimodal fusion, LLM integration, Firestore |

Project Advisor: Engr. Mehar Fatima (Lecturer, CSIT — NED University)  
Group Number:CT-22045 | Batch: 2022–2026


 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ANDROID APP (Capacitor)               │
│   Dashboard │ Chat │ Journal │ Activities │ To-Do        │
└──────────────────────┬──────────────────────────────────┘
                       │  HTTP (REST API)
┌──────────────────────▼──────────────────────────────────┐
│              FastAPI Backend (Python / Render.com)       │
│                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ RuleEngine  │──▶│PromptBuilder │──▶│  Groq LLaMA3 │  │
│  └──────┬──────┘   └──────────────┘   └──────────────┘  │
│         │                                                │
│  ┌──────▼──────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ScoreManager │   │CrisisDetector│   │SafetyFilter  │  │
│  └──────┬──────┘   └──────────────┘   └──────────────┘  │
│         │                                                │
│  ┌──────▼──────────────────────────────────┐            │
│  │         PyTorch FusionModel              │            │
│  │  Text Branch (MiniLM BERT → 256d)        │            │
│  │  Audio Branch (MFCC → 80d → 256d)        │            │
│  │  Attention Fusion → 512d → Score [0–1]   │            │
│  └──────────────────────────────────────────┘            │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Firebase Firestore (NoSQL Database)         │
│   Sessions │ Messages │ Risk Scores │ Crisis Logs        │
└─────────────────────────────────────────────────────────┘
```




 🤖 AI / ML System

Sentio uses a three-layer hybrid intelligence system:

Layer 1 — Keyword Heuristic (Real-time, ~0ms)
Rule-based scoring from clinically informed keyword sets. HIGH keywords (hopeless, worthless) → score ≥ 0.75. Smoothed via exponential moving average: `0.6 × new + 0.4 × previous`.

Layer 2 — Custom Deep Learning FusionModel (Background, every 5th message)

Trained on the DAIC-WOZ dataset (Distress Analysis Interview Corpus — Wizard of Oz, USC) with PHQ-8 depression labels.

```
Text Input ──▶ MiniLM (all-MiniLM-L6-v2) ──▶ 384d ──▶ Linear ──▶ 256d ──▶ 512d ──┐
                                                                                    ├──▶ Attention Gate ──▶ Weighted Fusion ──▶ Predictor ──▶ sigmoid ──▶ [0–1]
Audio Input ──▶ MFCC (librosa, 40 coeff) ──▶ 80d ──▶ Linear ──▶ 256d ──▶ 512d ──┘
```

Features extracted from audio:MFCC (40 coefficients), pitch, jitter, shimmer, spectral centroid, spectral rolloff, chroma features, speech energy — selected via ANOVA F-statistic feature selection.

Fusion: Attention-weighted fusion — the gate dynamically learns how much to trust text vs audio per input, enabling the model to detect the clinical disconnect between verbal content and vocal affect.

Training config:Adam optimizer, Binary Cross-Entropy loss, Dropout regularization, gradient norm clipping, learning rate scheduling, early stopping.

Blending: `score = 0.7 × ML score + 0.3 × heuristic` after first ML run.

Layer 3 — LLaMA 3 via Groq API (Response Generation)
The LLM is used only for generating empathetic conversational responses — not for scoring. The system prompt is dynamically constructed based on the current risk tier, enforcing tone, sentence limits, and safety boundaries defined in `sentio_boundaries.json`.



Experimental Results

Training and evaluation performed on the DAIC-WOZ dataset using Python, PyTorch, and scikit-learn.

| Metric | Value |
|---|---|
| Validation MAE | ~4.4 |
| Validation RMSE | ~5.4 |
| Dev Pearson Correlation | ~0.25 |
| Dev R² Score (peak) | ~0.03–0.25 |
| Crisis Detection Recall | ~100% (rule-based, deterministic) |

Training observations: Overfitting begins around epoch 5. Training loss converges to near-zero across 20 epochs. Validation loss plateaus early — indicating the DAIC-WOZ dataset's small size (~189–220 sessions) as the primary constraint on generalization. The hybrid heuristic + ML blending strategy compensates for this in production.

> Note: The ML model calibrates LLM tone and is not used as a standalone clinical diagnostic. The keyword heuristic provides real-time responsiveness while the ML score provides periodic calibration.



 App Screenshots
| Low Risk | Moderate Risk | High Risk |
|---|---|---|

<img width="220"  alt="image" src="https://github.com/user-attachments/assets/78922957-ec16-4e0a-983d-6cbbde52b40e" />

<img width="220" alt="image" src="https://github.com/user-attachments/assets/ffa8810d-ec45-424b-b942-3936f4c111bf" />

<img width="220"  alt="WhatsApp Image 2026-06-17 at 2 39 45 PM" src="https://github.com/user-attachments/assets/3f1798ed-e15a-427c-8ad0-e394e88d0136" />



🚨 Crisis Detection — Safety First

When Sentio detects crisis language, it bypasses the LLM entirely and immediately
returns a handcrafted empathetic response along with localized emergency contacts.
This is deterministic — 17 compiled regex patterns, ~100% recall, no AI guessing.


| Crisis Response | Emergency Contacts |
|---|---|

<img width="220"  alt="WhatsApp Image 2026-06-17 at 2 39 01 PM" src="https://github.com/user-attachments/assets/210a9371-a2d5-4b75-96c9-b0c9d71e4c8f" />

<img width="220" alt="WhatsApp Image 2026-06-17 at 2 38 38 PM" src="https://github.com/user-attachments/assets/b7b39aa9-8bb7-40be-8379-d4ca945b1371" />


> User typed "i want to die" — Sentio immediately responded with human warmth and
> surfaced Pakistan-specific helplines (Umang Helpline, Rozan Counseling, Emergency 1122)
> without any LLM involvement.



 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Mobile Frontend | Capacitor 8 + Vanilla JS | Android hybrid app wrapper |
| UI | HTML / CSS / Lucide Icons | Single-page app, dark mode |
| PDF Export | jsPDF | Export chat history as PDF |
| Backend Framework | FastAPI (Python) | Async REST API server |
| ASGI Server | Uvicorn | Production server |
| Data Validation | Pydantic v2 | Request/response models |
| LLM | LLaMA 3 (8B) via Groq | Empathetic response generation |
| Speech-to-Text | Whisper via Groq | Voice input transcription |
| HTTP Client | httpx | Async API calls |
| Deep Learning | PyTorch | FusionModel (text + audio) |
| Text Embeddings | sentence-transformers (MiniLM) | 384-d sentence embeddings |
| Audio Features | librosa | MFCC, pitch, jitter, shimmer |
| Logging | Loguru | Structured backend logs |
| Database | Firebase Firestore | NoSQL cloud storage |
| Authentication | Firebase Auth | Email + Google Sign-In |
| Deployment | Render.com | Backend cloud hosting |
| Dataset | DAIC-WOZ (USC) | Clinical depression corpus (PHQ-8) |



 Project Structure

```
sentio/
├── sentio_backend/
│   ├── main.py                      FastAPI entry point, CORS, routers
│   ├── requirements.txt
│   ├── api/
│   │   ├── chat.py                  POST /chat, POST /transcribe
│   │   └── sessions.py              Session CRUD endpoints
│   ├── chatbot/
│   │   ├── llm_interface.py         Abstract LLM base class
│   │   ├── api_llm.py               Groq/OpenAI-compatible adapter
│   │   ├── prompt_builder.py        Dynamic system prompt by risk level
│   │   └── safety_filter.py         Strip markdown, forbidden phrases
│   ├── core/
│   │   ├── config.py                Pydantic settings (env vars)
│   │   └── crisis_detection.py      Regex crisis patterns + hotlines
│   ├── models/
│   │   └── depression_model.py      PyTorch FusionModel (BERT + MFCC)
│   ├── services/
│   │   ├── chat_service.py          Orchestrates engine + DB write
│   │   ├── rule_engine.py           7-step message pipeline
│   │   ├── context_builder.py       Conversation memory (deque)
│   │   ├── risk_adapter.py          float score → LOW/MODERATE/HIGH
│   │   └── score_manager.py         Heuristic + background ML scoring
│   ├── database/
│   │   └── firestore_client.py      Firebase Admin SDK wrapper
│   └── config/
│       └── sentio_boundaries.json   Identity rules, forbidden phrases, crisis contacts
└── sentio_frontend/
    ├── www/
    │   ├── index.html              # Single HTML, all screens
    │   ├── app.js                  # All JS (~3000 lines), SPA navigation
    │   └── style.css               # Dark mode, chat bubbles, animations
    ├── android/                    # Capacitor Android project
    └── package.json                # Capacitor plugins
``` 


 Safety & Ethics

- Not a diagnostic tool — Sentio provides emotional awareness and support, not clinical diagnosis.
- Crisis detection uses 17 compiled regex patterns for suicidal/self-harm language. When triggered, the LLM is bypassed entirely and localized crisis hotlines are returned immediately.
- PHQ-8 aligned — risk thresholds (LOW / MODERATE / HIGH) map conceptually to PHQ-8 severity ranges.
- Data privacy — only extracted features are stored; raw audio is never persisted.
- UN SDG 3 — Good Health and Well-being.
- Plagiarism: 15% overall similarity (Turnitin) — within acceptable academic limits.



 Future Enhancements

- Real-time streaming voice analysis via WebSocket
- Multilingual support (Urdu and regional languages)
- IoT wearable integration (heart rate variability, sleep data)
- Clinical pathway integration with licensed psychologists
- Larger, more diverse training datasets beyond DAIC-WOZ



 License

MIT License 


<div align="center">

Made at NED University of Engineering & Technology, Karachi — June 2026

"Sentio is a positive step towards using AI for social benefit in the area of mental health."

</div>
