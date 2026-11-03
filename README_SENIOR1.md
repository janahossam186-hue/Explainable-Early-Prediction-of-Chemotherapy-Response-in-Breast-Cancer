# Explainable Early Prediction of Chemotherapy Response in Breast Cancer

Multimodal, longitudinal AI system for early prediction of **pathological complete response (pCR)** vs **non-pCR** to **neoadjuvant chemotherapy (NAC)** in breast cancer. The system fuses **MRI** and **blood tests** acquired at two time points (**T0**: pretreatment, **T1**: after 1–2 cycles), outputs pCR / non-pCR probabilities, and explains each prediction with **Grad-CAM** (MRI) and **SHAP** (blood, fusion). It is served through a secure REST backend and a clinician-facing dashboard, with adversarial / out-of-distribution (OOD) input flagging.

**Senior Project I – CSAI 490 · New Giza University · 2026** · Advisor: Dr. Laila Hesham

> ⚠️ An early-warning tool that supports clinicians. It does not prescribe treatment.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Objectives and Success Criteria](#objectives-and-success-criteria)
- [System Framework](#system-framework)
  - [Data Processing](#data-processing)
  - [MRI Branch (STI Architecture)](#mri-branch-sti-architecture)
  - [Blood Branch](#blood-branch)
  - [Fusion and Prediction Head](#fusion-and-prediction-head)
  - [Explainability and Output](#explainability-and-output)
- [Data](#data)
- [Evaluation Protocol](#evaluation-protocol)
- [Functional Requirements](#functional-requirements)
- [Non-Functional Requirements](#non-functional-requirements)
- [User Stories](#user-stories)
- [Project Plan](#project-plan)
- [Risk Log](#risk-log)
- [Team](#team)

---

## Problem Statement

1. NAC fails in some patients, but doctors only discover this at surgery, which is too late.
2. Early prediction would allow faster treatment changes.
3. MRI can detect tumour changes non-invasively after the first cycles.
4. Existing studies are small, single-timepoint, and poorly validated.
5. **This project:** a multi-timepoint MRI model (pre + mid-treatment), validated on unseen data.

The overall pCR rate after NAC is 19–30%.

**Guiding question:** can early imaging and clinical or biological data identify at-risk patients before NAC is completed? Predictions must be interpretable and clinically useful, and the system must protect sensitive patient information.

## Objectives and Success Criteria

| # | Phase | Objective | Target |
|---|---|---|---|
| O1 | Phase 1 | **MRI-based response prediction:** model using pretreatment and early-treatment MRI, where available, to predict pCR vs non-pCR. Evaluate precision, recall, and the precision–recall curve. | AUC **> 0.923** on an independent test set |
| O2 | Phase 2 | **MRI and blood-sample integration:** multimodal model combining MRI-derived features with blood-sample measurements, compared with MRI-only and blood-only models on precision, recall, and PR-AUC. | **+0.01 AUC** minimum gain over the best single-modality model |
| O3 | Phase 3 | **Model evaluation and explainability:** patient-level splits; AUC, precision, recall, F1-score, specificity, PR-AUC, and calibration. Present top features with an explainability method, where supported. | **Top 5** most influential MRI features and blood-sample measurements |
| O4 | End of project | **Enhance radiologists' prediction:** decision-support interface combining AI pCR / non-pCR probabilities with radiologists' assessments, showing probabilities and main contributing features. Compare with vs without AI on the test set. | **≥ 15%** relative reduction in radiologists' misclassification rate |

## System Framework

Both branches measure change between the same two time points: **T0** (before chemotherapy) and **T1** (after 1–2 cycles).

```
                         Dataset collection
                                 │
                 Quality checks and patient matching
                 ┌───────────────┴────────────────────────┐
                 ▼                                        ▼
 ┌──── MRI branch: STI architecture ────┐   ┌──────── Blood branch ────────┐
 │  T0 MRI (pretreatment)               │   │ Blood tests at T0, T1        │
 │  T1 MRI (early-cycle)                │   │ (pretreatment + early-cycle) │
 │        │                             │   │        │                     │
 │  MRI preprocessing                   │   │ Patient-level alignment      │
 │  (scanner harmonization)             │   │ (sync to MRI dates, δt)      │
 │        │                             │   │        │                     │
 │  Automatic 3D modeling               │   │ Blood preprocessing          │
 │        │                             │   │ (missing values,             │
 │  Habitat imaging                     │   │  normalization)              │
 │  (whole tumor + tissue subregions)   │   │        │                     │
 │     ┌──┴───────────┐                 │   │ Feature engineering          │
 │  Encoder T0     Encoder T1           │   │ (Δ = T1 − T0 per marker)     │
 │  (MedicalNet)   (same weights)       │   │        │                     │
 │     │              │                 │   │ Shared MLP encoder           │
 │  Region fusion  Region fusion        │   │ (same weights for T0, T1, Δ) │
 │  (subregions → one)                  │   │        │                     │
 │     └──────┬───────┘                 │   │ Blood feature vector         │
 │  Change vectors: T1 − T0             │   │ (T0 + T1 + Δ embeddings)     │
 │  (whole tumor and habitats)          │   └────────┬─────────────────────┘
 │        │                             │            │
 │  Temporal attention, δt = t1 − t0    │            │
 │  (change over the scan interval)     │            │
 │        │                             │            │
 │  MRI feature vector                  │            │
 │  (T0 + T1 + changes)                 │            │
 └────────┬─────────────────────────────┘            │
          └──────────────────┬───────────────────────┘
                             ▼
     Fusion module (cross-attention or concat + MLP)
                             │
     Prediction head (fully connected layers + softmax)  ◄── pCR labels (supervision)
                     ┌───────┴────────┐
                     ▼                ▼
               pCR (e.g. 0.72)   Non-pCR (e.g. 0.28)
                     └───────┬────────┘
                             ▼
   Explainability: Feature importance — Grad-CAM (MRI), SHAP (blood, fusion)
                   Key drivers — tumor region, biomarkers, Δ change
                             │
              Results dashboard (per-patient view)
                             │
       Clinician-facing output (probability and evidence)
```

### Data Processing

- **Dataset collection** → **quality checks and patient matching** before the data is split into the two branches.

### MRI Branch (STI Architecture)

| Step | Description |
|---|---|
| Inputs | T0 MRI (pretreatment) and T1 MRI (early-cycle) |
| MRI preprocessing | Scanner harmonization |
| Automatic 3D modeling | 3D tumour modelling from the MRI volume |
| Habitat imaging | Whole tumour + tissue subregions |
| Encoders T0 / T1 | MedicalNet encoder; T1 encoder uses the same weights as T0 |
| Region fusion | Per time point, subregions → one representation |
| Change vectors | T1 − T0, for the whole tumour and the habitats |
| Temporal attention | δt = t1 − t0; models change over the scan interval |
| Output | MRI feature vector (T0 + T1 + changes) |

### Blood Branch

| Step | Description |
|---|---|
| Inputs | Blood tests at T0 and T1 (pretreatment + early-cycle) |
| Patient-level alignment | Sync to MRI dates, δt |
| Blood preprocessing | Missing values, normalization |
| Feature engineering | Δ = T1 − T0 per marker |
| Shared MLP encoder | Same weights for T0, T1, Δ |
| Output | Blood feature vector (T0 + T1 + Δ embeddings) |

### Fusion and Prediction Head

- **Fusion module:** cross-attention, or concatenation + MLP, over the MRI and blood feature vectors.
- **Prediction head:** fully connected layers + softmax, supervised with pCR labels.
- **Output:** P(pCR) and P(non-pCR), which sum to 1.

### Explainability and Output

- **Feature importance:** Grad-CAM for MRI; SHAP for blood and fusion.
- **Key drivers:** tumour region, biomarkers, Δ change.
- **Results dashboard:** per-patient view.
- **Clinician-facing output:** probability and evidence.

## Data

### Exploratory data analysis: TCGA-BRCA breast imaging (TCIA)

Source: TCIA, TCGA-BRCA Images; full collection, CC BY 3.0.

| Property | Value |
|---|---|
| Size | ≈ 88 GB |
| Participants / studies / series / images | 139 / 164 / 1,877 / 229,286 |
| Scan years | 2000–2006 |
| Modalities | MRI: 1,858 series (137 patients); mammography: 19 series (5 patients) |
| MRI patients by main series type | DCE 130 · T2/STIR 128 · Diffusion 20 |
| Patients by number of studies | 1 study: 118 · 2+ studies: 21 |
| Series per study | median 11 (range 1–28) |
| Sites / scanners | 7 sites; GE, Siemens, Philips MRI |

**Quality issues**

- **Missing:** Study Description in 20 studies; Series Description in 34 series (3 patients).
- **No annotations:** no tumour segmentations.
- **Naming:** the same scan types appear under many names across sites.
- **Mixed protocols:** 4 vendors, 7 sites, 12 years; 1–2,000 slices per series.
- **Non-original:** ≈ 44% of series are localizers, calibration, or CAD-derived; 23 are screen captures.

**Takeaway:** only 21 patients have 2+ studies and there are no segmentations, so change-over-time modelling and tumour-based features need extra work. Non-original series must be filtered out first.

### Target data

| Priority | Source | Notes |
|---|---|---|
| 1 | Local Egyptian hospital data (e.g. Baheya) | Most desirable; collection deadline early December |
| 2 | I-SPY 1 and I-SPY 2 | Backup if local data does not arrive |

Design choices for the data:

- MRI and blood tests come from the **same patients** and are matched before training.
- The project targets datasets with **multiple MRI scans across the NAC process**.
- **Other MRI types** are added where available.
- **Subtype and treatment differences** are accounted for where the data permit.

## Evaluation Protocol

- **Splits and test set:** patient-level splits, with evaluation on an **independent, unseen test set**.
- **Metrics:** AUC, precision, recall, F1-score, specificity, PR-AUC, and calibration. Sensitivity and ROC-AUC are also reported on the test set.
- **Baselines:** the fused model is benchmarked against **single-modality (MRI-only, blood-only)** and **clinical-only** baselines.
- **Radiologist comparison:** radiologists' performance is compared with vs without AI support on the test set.
- **Explainability review:** explanations are generated for the test set and checked by a radiologist.

## Functional Requirements

| ID | Aspect | Requirement | Description | Validation |
|---|---|---|---|---|
| FR-01 | AI Engine | Longitudinal Feature Analysis | Identify changes in MRI and blood-derived features between pretreatment and early-treatment measurements. | Recompute the change features by hand on 3 patients and compare. |
| FR-02 | AI Engine | Modality-Specific Model Training | Train separate AI models to learn patterns associated with treatment response from MRI data and blood-sample data. | Each model trains on its own and its validation metrics are logged. |
| FR-03 | AI Engine | Multimodal Fusion | Combine MRI and blood-derived information across both time points to estimate the probability of achieving pCR. | Compare fused and single-modality models on the same split and report the difference. |
| FR-04 | AI Engine | Treatment-Response Prediction | Output the estimated probability of pCR and of non-pCR based on the available input data. | pCR and non-pCR probabilities sum to 1 for every test case. |
| FR-05 | AI Engine | Prediction Explainability | Provide supporting evidence through relevant MRI regions and influential blood-derived features. | Heatmaps and feature scores exist for every prediction; reviewed with a radiologist. |
| FR-06 | Software | RESTful API Data Ingestion & Routing | Secure REST API endpoints to receive, parse and route uploaded NIfTI/DICOM imaging files and CSV clinical records. | Each upload is parsed, saved to the right place and returns the expected status code; requests without a valid login or token are rejected. |
| FR-07 | Software | Relational Database Schema and ORM Management | ORM connected to a relational database storing patient demographic records, file storage paths, and historical inference logs, with strict foreign key constraints. | Run the migrations, then insert test patients. |
| FR-08 | Software | Client-Side Responsive Dashboard Interface | Responsive web interface to upload files, trigger server requests, and inspect patient summary cards and result tables. | Complete the full flow (upload → request → summary cards and results table) on desktop, tablet and phone screen sizes. |
| FR-09 | Security | Authentication and Role-Based Access | Authenticate every user and enforce role-based permissions (clinician, researcher, admin) on uploads, results, and exports; sessions expire. | Missing login or wrong role is denied; tokens expire. |
| FR-10 | Security | Adversarial and OOD Input Flagging | Flag inputs that look adversarial or far from the training data and route them to clinician review instead of a confident score. | Detection rate on crafted and shifted inputs, plus false-alarm rate on clean data. |

## Non-Functional Requirements

| ID | Aspect | Requirement | Description | Validation |
|---|---|---|---|---|
| NFR-1 | AI Engine | Response Time | Generate a prediction within an acceptable time after all required inputs are processed. | Time predictions on sample cases against a team-set target. |
| NFR-2 | AI Engine | Explainability | Provide interpretable explanations when supported by the implemented methods. | Explanations generated for the test set and checked by a radiologist. |
| NFR-3 | AI Engine | Clinical Safety | Communicate that predictions support clinical review and do not independently determine treatment changes. | Every output screen carries the clinical-review notice. |
| NFR-4 | AI Engine | Model Evaluation | Evaluate with sensitivity, specificity, precision, F1-score, and ROC-AUC where applicable. | Report these metrics on the test set. |
| NFR-5 | AI Engine | Hospital System Integration | Architecture supports integration with hospital information systems if interfaces and permissions are available. | Architecture review documents the needed interface points and permissions. |
| NFR-6 | Security | Adversarial Robustness | Evaluate against evasion attacks on MRI and blood features; report clean vs adversarial AUC at set perturbation budgets. | Attack suite run on the baseline and change-based models. |
| NFR-7 | Security | Privacy | No patient identifiers in DICOM headers, logs, or model artifacts; test the model for membership-inference leakage. | Membership-inference test shows no meaningful leakage; header and log audit finds no identifiers. |
| NFR-8 | Security | Encryption | Encrypt data in transit (TLS) and at rest, covering the database and stored scans. | Traffic capture shows no plaintext; stored files are unreadable without the key. |
| NFR-9 | Software | Timeframe | Complete HTTP request handling, file parsing, and data retrieval within a reasonable timeframe under normal local testing. | Time 20+ upload, parse and retrieval requests; report median and slowest times against a target set in advance. |
| NFR-10 | Software | Error Handling | Handle invalid file formats, missing fields, or runtime exceptions without crashing the server process. | Send a wrong file type, a corrupted DICOM, a CSV with missing columns, an empty upload, and a deliberately triggered server error. |

## User Stories

- **US-01 · Radiologist (treatment response prediction and explainability):** as a radiologist, I want to view the AI-predicted treatment response based on the patient's available MRI examinations and blood biomarker data, along with an explanation of the prediction, so that I can understand the factors contributing to the prediction and identify patients who may be showing early signs of non-response to neoadjuvant chemotherapy.
- **US-02 · Oncologist (clinical assessment support):** as an oncologist, I want to review the AI-generated treatment response prediction and any non-response warnings, so that I can identify patients who may require further clinical assessment and make informed decisions about their care.

## Project Plan

Timeline: October – April.

| Task | Owner(s) |
|---|---|
| Identifying relevant papers | Sara Ihab (security angle), Jana Hossam, Jana El Metwalli, Hussein Shaaban |
| Identifying gaps | Sara Ihab (security angle), Jana Hossam, Jana El Metwalli, Hussein Shaaban |
| Understanding previous work & writing literature review | Sara Ihab (security angle), Jana Hossam, Jana El Metwalli, Hussein Shaaban |
| Local data collection | Whole team |
| Initial features | Jana El Metwalli, Jana Hossam |
| Initial interface | Hussein Shaaban |
| Rest of features | Jana Hossam, Jana El Metwalli |
| Interface completion | Hussein Shaaban |
| Security testing and code security assurance | Sara Ihab |

## Risk Log

| Risk | Detail | Plan |
|---|---|---|
| Local data may not arrive | Egyptian hospital data (e.g. Baheya) is the most desirable but hard to obtain. | Allow until early December to collect local data; otherwise use I-SPY 1 and I-SPY 2. |
| Thin security literature | The cybersecurity angle is less researched for this project, so pipeline-security papers are hard to find. | Study the literature on adversarial attacks against AI healthcare systems and apply it to this project. |
| MRIs are not routine procedures | Most MRI models use one pretreatment scan, but the strongest signal comes from how the tumour changes after the first scan. | Target datasets with multiple MRI scans across the NAC process. |
| XAI needs expert oversight | A radiologist must supervise the XAI outputs. | Find an expert radiologist and review the XAI outputs with them. |
| Hardware (GPU) requirements | The datasets are large and need GPU hardware to train and run the models. | — |

## Team

| Name | ID |
|---|---|
| Jana ElMetwalli | 202301914 |
| Sara Ihab | 202301226 |
| Jana Hossam | 202302710 |
| Hussein Shaaban | 202300713 |

**Advisor:** Dr. Laila Hesham, New Giza University

**Repository:** `github.com/janahossam186-hue/Explainable-Early-Prediction-of-Chemotherapy-Response-in-Breast-Cancer`
