# 📜 Historical Manuscript Digitization & NLP Pipeline

## 📌 Project Overview
This project is an end-to-end multimodal Machine Learning pipeline designed to digitize, clean, and extract structured data from unstructured 18th and 19th-century historical manuscripts (mixed PDFs and Images). 

It bridges the gap between **Computer Vision (VLM)** and **Natural Language Processing (NLP)** to automate the archival process.

## ⚙️ Pipeline Architecture

### Stage 1: Multimodal OCR & VLM Extraction
* **Challenge:** Historical documents are highly irregular, memory-intensive, and feature faded handwriting.
* **Solution:** Deployed Microsoft's lightweight **Florence-2-Large** Vision-Language Model. Bypassed cloud GPU memory constraints (OOM errors) using custom architectural patching and dynamic resolution scaling.
* **Result:** Successfully converted raw, unstructured `.pdf`, `.jpg`, and `.png` files into raw machine-readable text.

### Stage 2: NLP Cleaning & Post-Processing
* **Challenge:** OCR on cursive and 18th-century typesetting introduces phonetic anomalies and spacing errors.
* **Solution:** Engineered a Python-based regular expression cleaner to standardize whitespace and remove anomalous punctuation without destroying historical vocabulary.

### Stage 3: Named Entity Recognition (NER)
* **Challenge:** Manually reading the digitized text to find key historical data is inefficient.
* **Solution:** Integrated the **spaCy (`en_core_web_md`)** NLP engine to scan the cleaned texts and autonomously extract structured historical entities.
* **Result:** Generated a structured database (`.csv`) mapping documents to specific `PERSON` (Historical Figures), `GPE/LOC` (Locations), and `DATE` markers.

## 🛠️ Tech Stack
* **Languages:** Python
* **Vision / Deep Learning:** PyTorch, Hugging Face `transformers`, Microsoft `Florence-2-large`
* **NLP:** `spaCy`
* **Data Processing & Analytics:** `pandas`, `matplotlib`, `seaborn`, `pdf2image`, `PIL`

## 📊 Analytics
The final stage of the pipeline generates a visual analytics dashboard, automatically graphing the most frequently mentioned historical figures and geographic locations across the ingested manuscript batch.
