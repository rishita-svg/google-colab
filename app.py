import os
import torch
import glob
import pandas as pd
import spacy
import re
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch

def fixed_get_imports(filename):
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

print("Initializing VLM and NLP engines...")
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
nlp = spacy.load("en_core_web_md")

def clean_text(text):
    cleaned = re.sub(r'[-–—/_]{2,}', ' ', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

def extract_entities(text):
    doc = nlp(text)
    return [{"Text": ent.text.strip(), "Category": ent.label_} for ent in doc.ents if ent.label_ in {"PERSON", "GPE", "LOC", "DATE", "ORG"}]

def run_pipeline(source_dir, output_csv):
    all_results = []
    files = glob.glob(os.path.join(source_dir, "*"))
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        extracted_text = ""
        ext = filename.split('.')[-1].lower()
        try:
            if ext == 'pdf':
                pages = convert_from_path(file_path, dpi=300)
                for page in pages:
                    inputs = processor(text="<OCR>", images=page, return_tensors="pt").to("cuda", torch.float16)
                    generated_ids = model.generate(**inputs, max_new_tokens=1024)
                    extracted_text += processor.batch_decode(generated_ids, skip_special_tokens=True)[0] + " "
            elif ext in ['jpg', 'jpeg', 'png']:
                img = Image.open(file_path).convert("RGB")
                inputs = processor(text="<OCR>", images=img, return_tensors="pt").to("cuda", torch.float16)
                generated_ids = model.generate(**inputs, max_new_tokens=1024)
                extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            cleaned = clean_text(extracted_text)
            for ent in extract_entities(cleaned):
                all_results.append({"Document": filename, "Entity": ent["Text"], "Type": ent["Category"]})
        except Exception as e:
            print(f"Error on {filename}: {e}")
    
    pd.DataFrame(all_results).to_csv(output_csv, index=False)
    print(f"Pipeline complete. Saved to {output_csv}")

if __name__ == "__main__":
    run_pipeline("/content/manuscripts", "/content/final_report.csv")
