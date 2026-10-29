# 🌿 VETIVER Project — Team Structure & Operational Charter

**Institution:** Jomo Kenyatta University of Agriculture and Technology (JKUAT)  
**Project:** Vetiver Grass AI Bio-Engineering & Geotechnical Decision Platform  
**Overall Project Lead & Technical Lead:** Brian

---

## 1. Executive Summary & Team Topology

The project is structured into **two core functional divisions** comprising **6 team members**:
1. **Technical & Systems Division** (3 members) — Model optimization, platform engineering, API integrations (WhatsApp), and on-premise infrastructure.
2. **Data & Sanitization Division** (3 members) — Literature acquisition, empirical data mining, agronomic validation, schema standardization, and knowledge base enrichment.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     OVERALL PROJECT LEADER & TECHNICAL LEAD              │
│                                  (Brian)                                 │
│          • Architecture & Pipeline Integration • Repository Owner        │
│          • Model Sign-Off                      • JKUAT Deployment Lead   │
└───────────────────┬──────────────────────────────────┬───────────────────┘
                    │                                  │
     ┌──────────────▼──────────────┐    ┌──────────────▼──────────────┐
     │  TECHNICAL & SYSTEMS TEAM   │    │  DATA & SANITIZATION TEAM   │
     ├─────────────────────────────┤    ├─────────────────────────────┤
     │ • Brian (Technical Lead)    │    │ • Data Lead & QA Analyst    │
     │ • WhatsApp API Engineer     │    │ • Literature Sourcing Spec. │
     │ • Platform & MLOps Engineer │    │ • Agronomic Domain Spec.    │
     └─────────────────────────────┘    └─────────────────────────────┘
```

---

## 2. Role Assignments & Detailed Task Backlogs

### Role 1: Overall Project Lead & Principal AI/ML Architect (Brian)
* **Direct Reports:** WhatsApp Engineer, Platform Engineer, Data Lead.
* **Core Responsibilities:**
  * Define system architecture, model choices, and technical milestones.
  * Maintain repository master access, review and merge all GitHub Pull Requests.
  * Direct training of the core CatBoost model and LlamaIndex vector store.
  * Coordinate between Data and Technical teams to ensure schema parity.
  * Lead physical team review meetings and the final JKUAT hardware deployment session.
* **Deliverables:**
  - Architecture roadmap and repository hygiene.
  - Integration of CatBoost classifier with LlamaIndex RAG engine.
  - Final sign-off on models prior to JKUAT desktop production deployment.

---

### Role 2: WhatsApp API & Messaging Integration Engineer (Technical Team)
* **Reports To:** Brian (Technical Lead).
* **Core Responsibilities:**
  * Architect the WhatsApp bot interface so field engineers, farmers, and reviewers can query the system via mobile.
  * Implement webhook handlers using **FastAPI** to receive and parse incoming WhatsApp messages.
  * Route requests:
    * *Tabular parameter queries* $\rightarrow$ CatBoost suitability predictor.
    * *Natural language geotechnical questions* $\rightarrow$ LlamaIndex RAG engine.
  * Format outgoing WhatsApp responses (chunking long responses, bold headers, concise bullet points).
* **Key Tasks:**
  - [ ] Set up Meta WhatsApp Business Cloud API or Twilio WhatsApp sandbox.
  - [ ] Create `whatsapp_service.py` with webhook verification and HMAC security.
  - [ ] Implement conversation state tracking for multi-turn site assessment dialogues over WhatsApp.
  - [ ] Handle error states (e.g., user sends invalid parameters or unsupported media).

---

### Role 3: Platform & MLOps Engineer (Technical Team)
* **Reports To:** Brian (Technical Lead).
* **Core Responsibilities:**
  * Maintain Streamlit application styling, UI responsiveness, and real-time state.
  * Containerize the platform using Docker and Docker Compose.
  * Prepare the **JKUAT Desktop Workstation environment**:
    * Configure local Ollama daemon as a persistent background service (Windows NSSM or Linux systemd).
    * Optimize CPU thread pinning (`num_thread`) and GPU offloading (if dedicated NVIDIA hardware is present).
    * Configure local network tunneling (Cloudflare Tunnel or ngrok) for mobile/WhatsApp webhook routing.
* **Key Tasks:**
  - [ ] Build a lightweight FastAPI wrapper (`api.py`) exposing `/predict` and `/chat` endpoints.
  - [ ] Test container builds on the target JKUAT lab desktop machines.
  - [ ] Configure local persistent volumes for `storage/`, `data/`, and `knowledge_base/`.
  - [ ] Set up automated backup scripts for the vector index and model weights.

---

### Role 4: Data Lead & Quality Assurance Analyst (Data & Sanitization Team)
* **Reports To:** Brian (Project Lead).
* **Core Responsibilities:**
  * Supervise the online Google Drive / Google Sheets master dataset.
  * Enforce strict column schema, controlled vocabularies, and missing-value protocols.
  * Convert unstructured spreadsheet notes into clean, machine-readable CSVs.
  * Standardize geotechnical units:
    * Root depth $\rightarrow$ centimeters (`cm`).
    * Soil shear strength / cohesion $\rightarrow$ kilopascals (`kPa`).
    * Salinity $\rightarrow$ Electrical Conductivity ($dS/m$) or qualitative categorical scales.
* **Key Tasks:**
  - [ ] Audit the current 91-row master dataset and resolve inconsistent spellings (e.g. `"Semi- arid."` vs `"semi-arid"`).
  - [ ] Establish automated export from Google Sheets to `data/vetiver_growth_data.csv`.
  - [ ] Validate that all categorical columns match the definitions in `config.py`.
  - [ ] Perform deduplication and data leakage audits between training and validation splits.

---

### Role 5: Literature & Document Sourcing Specialist (Data & Sanitization Team)
* **Reports To:** Data Lead.
* **Core Responsibilities:**
  * Source peer-reviewed geotechnical journal papers, Vetiver Network International (TVNI) technical bulletins, and ICRAF/FAO field manuals.
  * Focus on East African / Kenyan soil environments (e.g., Kenyan Central Highlands red volcanic soils, Rift Valley sandy loams, black cotton clays).
  * Extract and clean text from PDFs (removing headers, footers, bibliographic noise).
  * Place processed documents into `knowledge_base/` with proper citation metadata.
* **Key Tasks:**
  - [ ] Source at least 15–20 high-quality technical PDFs on Vetiver root tensile strength, slope reinforcement, and erosion control.
  - [ ] Clean and segment documents into topical markdown/text files for vector embedding.
  - [ ] Verify that document chunks contain clear location and author citations for LlamaIndex source attribution.

---

### Role 6: Agronomic & Geotechnical Domain Specialist (Data & Sanitization Team)
* **Reports To:** Data Lead.
* **Core Responsibilities:**
  * Act as the subject-matter authority on Vetiver (*Chrysopogon zizanioides*) biological tolerances.
  * Design realistic failure boundaries and edge cases for CatBoost model training (e.g., maximum waterlogging tolerance, minimum temperature limits, bedrock refusal thresholds).
  * Review RAG chatbot answers for scientific validity and eliminate hallucinations.
  * Create a benchmark set of 30 "golden test questions" to evaluate model and RAG accuracy.
* **Key Tasks:**
  - [ ] Expand synthetic failure cases in `train_model.py` from 16 to 40+ realistic scenarios based on field mechanics.
  - [ ] Validate TreeSHAP feature attributions to ensure feature impact aligns with geotechnical physics.
  - [ ] Compile the golden evaluation benchmark dataset for testing chatbot responses during team reviews.

---

## 3. Governance, Reporting & Meeting Structure

### Reporting Hierarchy
* **Technical Engineers (WhatsApp, Platform)** $\rightarrow$ Report to **Brian** on technical architecture and code PRs.
* **Data Specialists (Literature, Agronomy)** $\rightarrow$ Report to **Data Lead**, who delivers verified data batches to **Brian**.
* **Data Lead** $\rightarrow$ Reports to **Brian** on dataset milestones, schema updates, and validation scores.

### Meeting Cadence
| Meeting Type | Frequency | Location | Focus |
| :--- | :--- | :--- | :--- |
| **Async Daily Standup** | Daily | WhatsApp Group / GitHub | 3 questions: What did you finish? What are you doing today? Any blockers? |
| **Weekly Technical Sync** | Once / week | JKUAT Tech Lab / Physical | Code review, PR merges, model metric evaluation, live demo testing. |
| **Bi-Weekly All-Hands** | Every 2 weeks | Physical Meeting | Cross-team alignment (Data presents new data; Tech demos updated system). |
| **Deployment Workshop** | Milestone Event | JKUAT Desktop Lab | **All 6 members present** for final on-premise installation and server configuration. |

---

## 4. GitHub Collaboration Protocol

### Branching Strategy
* **`main`**: Protected branch. Only deploys production code. Merges require PR approval from Brian.
* **`develop`**: Active integration branch.
* **Feature Branches**:
  * `tech/whatsapp-api` — WhatsApp webhook and integration work.
  * `tech/fastapi-backend` — API wrappers and server configurations.
  * `data/sheet-sanitization` — Dataset cleanups and schema updates.
  * `data/literature-sourcing` — New PDF text documents for `knowledge_base/`.
  * `model/catboost-tuning` — Hyperparameter tuning and expanded boundary training.

### Commit Conventions
All team members must follow standard semantic commits:
* `feat:` A new user-facing feature (e.g. `feat: add WhatsApp webhook handler`)
* `data:` Dataset additions or cleaning (e.g. `data: sanitize salinity units in growth csv`)
* `docs:` Documentation or research updates (e.g. `docs: add TVNI technical bulletin on root depth`)
* `fix:` Bug fixes (e.g. `fix: handle NoneType in streaming response generator`)
* `perf:` Speed optimizations (e.g. `perf: optimize Ollama CPU thread allocation`)

---

## 5. Deployment Roadmap (JKUAT Workstations & WhatsApp)

### Phase A: Data & Model Enrichment (Weeks 1–2)
- Data team audits Google Sheet and expands knowledge base with Kenyan case studies.
- Technical team builds FastAPI middleware and verifies model retraining pipeline.

### Phase B: WhatsApp Integration (Weeks 2–3)
- WhatsApp Engineer sets up Meta/Twilio sandbox and webhooks.
- Connects WhatsApp incoming prompts to `rag_engine.py` and `cb_model`.

### Phase C: On-Premises JKUAT Deployment (Week 4)
- **Physical Team Event at JKUAT:**
  - Setup high-performance JKUAT Desktop workstation (Ubuntu / Windows Server).
  - Configure background Ollama service daemon.
  - Deploy containerized platform via `docker-compose.yml`.
  - Expose webhook endpoint securely for WhatsApp API access.
  - End-to-end testing with patron, faculty reviewers, and all 6 team members.
