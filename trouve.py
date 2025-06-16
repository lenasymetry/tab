# === IMPORTS ===
import streamlit as st
from pdf2image import convert_from_bytes
from google.oauth2 import service_account
from google.cloud import vision
import io
from PIL import Image
import unicodedata
import re
import os
import json

# === INITIALISATION DU CLIENT GOOGLE VISION (Streamlit Secrets) ===
service_account_info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
credentials = service_account.Credentials.from_service_account_info(service_account_info)
client = vision.ImageAnnotatorClient(credentials=credentials)

# === CONVERSION DU PDF EN IMAGES ===
def pdf_to_images(pdf_bytes):
    return convert_from_bytes(pdf_bytes)

# === APPEL À L'OCR DE GOOGLE VISION POUR UNE IMAGE ===
def vision_ocr_detect_text(image_pil):
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    content = img_byte_arr.getvalue()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")

    annotations = response.text_annotations
    if not annotations:
        return []

    words = []
    for ann in annotations[1:]:
        vertices = ann.bounding_poly.vertices
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        words.append({"text": ann.description, "bbox": bbox})

    return words

# === REGROUPER LES MOTS EN LIGNES BASÉ SUR LEUR POSITION Y ===
def group_words_by_lines(words, y_tolerance=10):
    lines = []
    words_sorted = sorted(words, key=lambda w: w['bbox'][1])

    for w in words_sorted:
        x_min, y_min, x_max, y_max = w['bbox']
        mid_y = (y_min + y_max) / 2
        placed = False
        for line in lines:
            line_y = line['y_mean']
            if abs(mid_y - line_y) <= y_tolerance:
                line['words'].append(w)
                ys = [ (ww['bbox'][1] + ww['bbox'][3]) / 2 for ww in line['words'] ]
                line['y_mean'] = sum(ys) / len(ys)
                placed = True
                break
        if not placed:
            lines.append({'y_mean': mid_y, 'words': [w]})

    for line in lines:
        line['words'] = sorted(line['words'], key=lambda w: w['bbox'][0])

    return lines

# === SUPPRIMER LES ACCENTS ET MINUSCULES POUR COMPARAISON ===
def normalize(text):
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8').lower()

# === EXTRAIRE LE NOMBRE APRÈS UN MOT CLÉ ===
def extract_amount_after_keyword(line_text, keyword):
    normalized_text = normalize(line_text)
    normalized_keyword = normalize(keyword)

    if normalized_keyword in normalized_text:
        index = normalized_text.find(normalized_keyword) + len(normalized_keyword)
        substring = normalized_text[index:]
        match = re.search(r"([\d\s.,]+?)(?:\s|-)", substring)
        if match:
            value_str = match.group(1).replace(',', '.').replace(' ', '')
            try:
                return float(value_str)
            except:
                return None
    return None

# === INTERFACE STREAMLIT ===
st.set_page_config(page_title="OCR PDF – Recherche par mot", layout="centered")

# === AFFICHAGE DU LOGO ===
if os.path.exists("logo.png"):
    logo_img = Image.open("logo.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo_img)
else:
    st.warning("Logo non trouvé. Assure-toi que 'logo.png' est présent dans le même dossier que ce script.")

st.title("🔎 Recherche et extraction de montants après un mot dans un PDF")

uploaded_file = st.file_uploader("Dépose ton fichier PDF scanné ici", type=["pdf"])
search_word = st.text_input("Entrez le mot à rechercher (ex : échéance)", "").strip()

if uploaded_file and search_word:
    pdf_bytes = uploaded_file.read()
    try:
        images = pdf_to_images(pdf_bytes)
        st.success(f"{len(images)} page(s) analysée(s).")

        matching_lines = []
        total_amount = 0.0

        for pil_img in images:
            pil_img = pil_img.convert("RGB")
            words = vision_ocr_detect_text(pil_img)
            if not words:
                continue
            lines = group_words_by_lines(words)

            for line in lines:
                line_text = " ".join([w['text'] for w in line['words']])
                if normalize(search_word) in normalize(line_text):
                    amount = extract_amount_after_keyword(line_text, search_word)
                    matching_lines.append((line_text, amount))
                    if amount is not None:
                        total_amount += amount

        if matching_lines:
            st.subheader("📋 Lignes contenant le mot recherché :")
            for i, (txt, val) in enumerate(matching_lines):
                st.write(f"• {txt}")
                if val is not None:
                    st.write(f"   ➤ Montant détecté : **{val}** €")
                else:
                    st.write("   ⚠️ Aucun montant détecté après ce mot.")

            st.markdown(
                f"""
                <div style="background-color:#FFD700;padding:15px;border-radius:10px;text-align:center">
                    <h3 style="color:#000000;">💰 Somme totale détectée après '<em>{search_word}</em>' : <strong>{total_amount:.2f} €</strong></h3>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Aucune ligne contenant ce mot n’a été trouvée.")

    except Exception as e:
        st.error(f"Erreur : {e}")
