<div align="center">

# 🛡️ PII Shield

### Enterprise-Style Document Privacy & PII Redaction Platform

**Detect → Classify → Replace → Validate → Export**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![python-docx](https://img.shields.io/badge/python--docx-DOCX%20Engine-2B579A?style=for-the-badge&logo=microsoftword&logoColor=white)](https://python-docx.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Educational-lightgrey?style=for-the-badge)](#-license)
[![Status](https://img.shields.io/badge/Status-Active%20Development-success?style=for-the-badge)](#project-status)

</div>

---

PII Shield is a document privacy application designed to identify **Personally Identifiable Information (PII)** inside Microsoft Word (`.docx`) documents, replace detected values with consistent synthetic values, validate the sanitized document, and provide an interactive interface for reviewing the results.

The project combines a **Python redaction engine** with a **Streamlit web interface** so that document sanitization can be performed without manually editing the source document.

---

## 📑 Table of Contents

- [Why PII Shield?](#-why-pii-shield)
- [Key Features](#-key-features)
- [Interactive Web Application](#️-interactive-web-application)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [End-to-End Processing Flow](#-end-to-end-processing-flow)
- [Technology Stack](#️-technology-stack)
- [Installation](#️-installation)
- [Running the App](#️-run-the-application)
- [Testing](#-run-detector-tests)
- [Example Result](#-example-processing-result)
- [Privacy & Security Design](#-privacy--security-design)
- [Limitations](#️-limitations)
- [What Makes This Different](#-what-makes-this-project-different)
- [Future Enhancements](#-future-enhancements)
- [Deployment](#️-deployment)
- [Project Checklist](#-project-checklist)
- [Author](#-author)
- [License](#-license)

---

## ✨ Why PII Shield?

Sensitive information can appear throughout business documents — in paragraphs, tables, headers, and footers. Manually finding and removing these values is slow, inconsistent, and difficult to audit.

**PII Shield automates the workflow:**

```
DOCX Document
     │
     ▼
Document Structure Analysis
     │
     ▼
PII Detection
     │
     ├── Person / Name
     ├── Email
     ├── Phone
     ├── Address
     ├── DOB / Date
     ├── SSN
     ├── Credit Card
     ├── IP Address
     └── Company
     │
     ▼
Entity Classification
     │
     ▼
Synthetic Replacement
     │
     ▼
Redacted DOCX
     │
     ▼
Post-Redaction Validation
     │
     ▼
Interactive Report + Downloads
```

---

## 🚀 Key Features

### 📄 DOCX PII Detection

Upload a Microsoft Word document and analyze its contents for potentially sensitive information. The processing pipeline works across:

| Location | Supported |
|---|:---:|
| Paragraphs | ✅ |
| Tables | ✅ |
| Headers | ✅ |
| Footers | ✅ |

This matters because sensitive information isn't necessarily stored only in normal paragraph text.

### 🏷️ Multiple PII Categories

The detector layer supports pattern/context-based detection for the following categories:

| Category | Example |
|---|---|
| `PERSON` / Name | John Smith |
| `EMAIL` | john.smith@example.com |
| `PHONE` | +91 9876543210 |
| `DOB` / Date | 15/08/1990 |
| `SSN` | 123-45-6789 |
| `CREDIT_CARD` | 4111 1111 1111 1111 |
| `IP_ADDRESS` | 192.168.1.25 |
| `ADDRESS` | 123 Example Road, Pune |
| `COMPANY` | ABC Technologies Limited |

> Detection is built around **deterministic patterns and contextual heuristics** rather than requiring a large external AI model.

### 🔄 Consistent Synthetic Replacement

Detected values aren't simply deleted — PII Shield generates replacement values so the sanitized document stays readable.

**Example:**

```diff
- Original: John Smith contacted john.smith@example.com.
+ Redacted: Rahul Kapoor contacted rahul.kapoor@example.com.
```

The replacement approach preserves document usability while removing the original sensitive value.

### ✅ Post-Redaction Validation

A core part of the project is the **second scan**. After creating the redacted document, the system checks the sanitized result again.

```
Original Document
     │
     ▼
 PII Detection
     │
     ▼
 Replacement
     │
     ▼
Redacted Document
     │
     ▼
Second Detection Scan
     │
     ├── 0 remaining  → ✅ PASSED
     └── Remaining    → ⚠️ REVIEW
```

This makes the system more than a simple find-and-replace script.

---

## 🖥️ Interactive Web Application

PII Shield includes a **Streamlit interface** designed as an enterprise-style privacy dashboard.

### 📊 Overview

The main dashboard provides:

- DOCX upload
- File information
- Processing status
- PII count & category count
- Paragraph / table count
- Processing time
- Detection breakdown
- Validation status
- Redacted document download

### 🔍 Detection Explorer

A detailed view of detected entities:

| Type | Original | Replacement | Start | End |
|---|---|---|---:|---:|
| `EMAIL` | original@email.com | synthetic@email.com | 120 | 142 |
| `PHONE` | +91 XXXXXXXX | +91 XXXXXXXX | 350 | 363 |
| `COMPANY` | ABC Limited | XYZ Industries | 600 | 613 |

> Results can also be exported as **CSV**.

### ⚖️ Before / After

A direct, transparent comparison:

```
Original Value
      ↓
Synthetic Replacement
```

### ✅ Validation

Displays:

- Total detected entities
- Remaining entities
- Validation status
- Remaining entity records when manual review is required

### 🗒️ Audit Log

Records key processing events:

- Document uploaded
- Document structure analyzed
- PII detection completed
- Entities processed
- Synthetic replacements generated
- Post-redaction validation completed

---

## 🧠 System Architecture

```
                     ┌──────────────────────┐
                     │     Streamlit UI     │
                     │       app.py         │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │  Redaction Engine    │
                     │ redaction_engine.py  │
                     └──────────┬───────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
         ┌────────────────┐           ┌────────────────┐
         │ PII Detectors  │           │ DOCX Processor │
         │  detectors.py  │           │                │
         └───────┬────────┘           └───────┬────────┘
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                     ┌──────────────────────┐
                     │ Synthetic Replacement│
                     └──────────┬───────────┘
                                ▼
                     ┌──────────────────────┐
                     │ Validation / Re-scan │
                     └──────────┬───────────┘
                                ▼
                     ┌──────────────────────┐
                     │ Redacted DOCX + JSON │
                     └──────────────────────┘
```

---

## 📁 Project Structure

```
PII-Redaction-Tool/
│
├── app.py                     # Streamlit UI
├── main.py                    # CLI entry point
├── redaction_engine.py        # Core processing engine
├── detectors.py                # PII detection logic
├── evaluator.py                # Evaluation / reporting
├── test_detectors.py           # Detector test suite
├── requirements.txt
├── README.md
├── .gitignore
│
├── input/
│   └── sample.docx
│
└── output/
    ├── redacted_prospectus.docx
    └── detection_report.json
```

### 📌 File Responsibilities

<details>
<summary><b>🎨 <code>app.py</code> — Streamlit presentation layer</b></summary>

Responsible for:
- Upload interface
- Navigation
- Dashboard
- Detection Explorer
- Before / After review
- Validation report
- Audit log
- Downloads & visual styling

Calls the redaction engine instead of implementing the core redaction logic itself.
</details>

<details>
<summary><b>⌨️ <code>main.py</code> — Command-line execution layer</b></summary>

Useful when the application needs to run without the Streamlit UI.

```
Input DOCX → main.py → Redaction → Output DOCX → Detection report
```
</details>

<details>
<summary><b>⚙️ <code>redaction_engine.py</code> — Core document-processing layer</b></summary>

Connects:
- DOCX parsing
- Detection
- Entity processing
- Replacement
- Validation
- Output generation

The Streamlit app calls `redact_docx_bytes(...)` to process an uploaded document.
</details>

<details>
<summary><b>🔎 <code>detectors.py</code> — Detection layer</b></summary>

Contains the logic/patterns used to identify PII categories:
- Regular expressions
- Pattern validation
- Contextual heuristics
- Entity classification
</details>

<details>
<summary><b>📈 <code>evaluator.py</code> — Evaluation / reporting utility</b></summary>

Used to evaluate detection output and validate generated reports.
</details>

<details>
<summary><b>🧪 <code>test_detectors.py</code> — Detector verification script</b></summary>

Checks whether the detector layer identifies representative PII examples such as:

```
John Smith
john.smith@example.com
+91 9876543210
15/08/1990
123-45-6789
4111 1111 1111 1111
192.168.1.25
123 Example Road, Pune
ABC Technologies Limited
```
</details>

---

## 🔄 End-to-End Processing Flow

| Step | Description |
|---|---|
| **1. Upload** | User uploads `document.docx` through the Streamlit interface |
| **2. Read the document** | The engine reads the DOCX structure — paragraphs, tables, headers, footers |
| **3. Detect** | Detector patterns (`EMAIL_PATTERN`, `PHONE_PATTERN`, etc.) search for PII |
| **4. Classify** | Each detected value becomes an entity with `Type`, `Original Value`, `Start`, `End` |
| **5. Replace** | A synthetic replacement is generated per entity type |
| **6. Create redacted doc** | Original DOCX → sanitized `PII_Shield_Redacted_<name>.docx` |
| **7. Validate** | Sanitized content is scanned again → `PASSED` if `remaining_count == 0`, else `REVIEW` |
| **8. Generate reports** | Redacted DOCX, detection results, validation result, audit info, CSV export |

**Replacement pipeline:**

```
Original Entity
      │
      ▼
  Entity Type
      │
      ▼
Replacement Generator
      │
      ▼
Synthetic Entity
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| 🐍 **Python** | Core application language |
| 🎈 **Streamlit** | Interactive web interface |
| 🐼 **Pandas** | Result processing & tabular data |
| 📊 **Altair** | Detection visualization |
| 📝 **python-docx** | DOCX document processing |
| 🔤 **Regular Expressions** | Pattern-based PII detection |
| 🗂️ **JSON** | Detection / report serialization |
| ✅ **PyTest / Python scripts** | Detector verification |

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PII-Redaction-Tool.git
cd PII-Redaction-Tool
```

> Replace `YOUR_USERNAME` with your GitHub username.

### 2️⃣ Create a virtual environment

**Windows**

```powershell
python -m venv venv
```

Activate:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit interface:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, normally:

```
http://localhost:8501
```

### 💻 Run Without the UI

For command-line processing:

```bash
python main.py
```

The CLI workflow generates:

```
output/redacted_prospectus.docx
output/detection_report.json
```

---

## 🧪 Run Detector Tests

```bash
python test_detectors.py
```

A successful detector test reports the expected entity categories and a total number of detected test entities.

---

## 📊 Example Processing Result

For a large DOCX document, the application can report statistics such as:

| Metric | Value |
|---|---:|
| PII Detected | 343 |
| PII Types | 5 |
| Paragraphs | 1006 |
| Tables | 76 |
| Processing Time | X.X s |

> The exact values depend on the uploaded document and detector rules.

---

## 🔐 Privacy & Security Design

PII Shield is designed around a simple principle:

> **Do not expose the original sensitive value unnecessarily.**

The application focuses on:

- ✅ Deterministic detection
- ✅ Synthetic replacement
- ✅ Local document processing
- ✅ Post-redaction validation
- ✅ Auditable processing results

> For production deployment, additional controls should be added before handling real confidential documents.

---

## ⚠️ Limitations

PII detection is **not** equivalent to perfect privacy assurance. Pattern-based systems can produce:

**False positives** — a value may match a pattern but not actually be PII.
> Example: `123-45-6789` may match an SSN pattern even when used as dummy data.

**False negatives** — PII may be missed when:
- Formatting is unusual
- Information is embedded in images
- Text is represented as shapes
- The document uses uncommon structures
- A value does not match an existing detector pattern

> Therefore, **"Validation PASSED"** should be interpreted as *"No remaining PII was detected by the implemented detection rules"* — not a mathematical guarantee that every sensitive value has been removed.

---

## 🌟 What Makes This Project Different?

Instead of being just:

> ~~"Regex that replaces PII"~~

PII Shield is structured as a **complete privacy workflow**:

```
Detection + Classification + Consistent Synthetic Replacement
    + Validation + Auditability + Interactive Review + Export
```

The key differentiator is the **post-redaction validation loop** — the system does not simply assume replacement succeeded. **It checks again.**

---

## 🚀 Future Enhancements

| Feature | Description |
|---|---|
| 🎯 **Confidence Scores** | e.g. `EMAIL 99%`, `PHONE 98%`, `ADDRESS 86%`, `PERSON 78%` to prioritize manual review |
| 🖼️ **OCR-Based PII Detection** | Image → OCR → Extracted Text → PII Detection → Redaction |
| 📄 **PDF Support** | Extend from DOCX to PDF, TXT, CSV, XLSX |
| 🧩 **Custom PII Rules** | Org-specific patterns like `EMPLOYEE_ID`, `CUSTOMER_ID`, `POLICY_NUMBER` |
| 🛡️ **Redaction Policies** | Configurable `Strict` / `Balanced` / `Minimal` policies |
| 📈 **Analytics Dashboard** | Track documents processed, PII trends, pass rate, avg. processing time |
| 📤 **Enterprise Audit Export** | Generate JSON / CSV / PDF audit reports for compliance |

---

## ☁️ Deployment

```
GitHub Repository
        │
        ▼
Cloud Deployment
        │
        ▼
Streamlit Application
        │
        ▼
Public / Restricted URL
```

Before deploying confidential documents, review:

- [ ] Data retention
- [ ] Uploaded-file handling
- [ ] Application logs
- [ ] Access control
- [ ] Authentication
- [ ] Storage configuration
- [ ] Organization privacy requirements

> ⚠️ **Do not use a public deployment for real confidential documents without appropriate security controls.**

---

## 🧹 .gitignore

```gitignore
# Virtual environment
venv/
.venv/

# Python
__pycache__/
*.py[cod]

# Streamlit
.streamlit/secrets.toml

# Generated output
output/*
!output/.gitkeep

# Local input documents
input/*
!input/.gitkeep

# Environment files
.env
.env.*

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

> ⚠️ **Important:** Do not commit real confidential documents to GitHub.

---

## 📦 requirements.txt

```txt
streamlit
pandas
altair
python-docx
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🔬 Testing Strategy

| Level | Command |
|---|---|
| Detector testing | `python test_detectors.py` |
| CLI processing | `python main.py` |
| UI testing | `streamlit run app.py` |

Then verify: DOCX upload → Analyze button → Detection Explorer → Before/After → Validation → Audit Log → Redacted DOCX download → CSV download.

---

## 📋 Project Checklist

- [x] `app.py` runs
- [x] `main.py` runs
- [x] Detector tests pass
- [x] DOCX upload works
- [x] PII detection works
- [x] Synthetic replacement works
- [x] Redacted DOCX opens correctly
- [x] Validation page works
- [x] Detection Explorer shows results
- [x] Before / After shows replacements
- [x] Audit Log records processing
- [x] DOCX download works
- [x] CSV download works
- [x] `requirements.txt` is updated
- [x] `.gitignore` exists
- [x] No confidential input documents are committed
- [x] README is included
- [x] Repository contains only final project files

---

## 🧩 Project Philosophy

```
Identify → Understand → Transform → Verify → Report
```

The goal is not merely to hide text. The goal is to create a **repeatable, inspectable, and verifiable document sanitization workflow**.

---

## 👨‍💻 Author

**Mouli Banerjee**
B.Tech Computer Science & Engineering — Data Science / Data Analytics Focus

---

## 📄 License

This project is intended for **educational, portfolio, and demonstration purposes** unless a separate license is provided.

---

<div align="center">

### ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub and sharing feedback or improvements!

---

**Project Status**

| Field | Value |
|---|---|
| Current Version | `v1.0` |
| Status | 🟢 Active Development |
| Primary Interface | Streamlit |
| Primary Document Format | Microsoft Word DOCX |

```
Upload → Detect → Classify → Replace → Validate → Review → Export
```

</div>
