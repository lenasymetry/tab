# === IMPORTS ===
import streamlit as st
from pdf2image import convert_from_bytes
from google.oauth2 import service_account
from google.cloud import vision
import io
import re
from PIL import Image, ImageDraw, ImageFont
import unicodedata
import os
import json

# === INITIALISATION DU CLIENT GOOGLE VISION (Streamlit Secrets) ===
service_account_info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
credentials = service_account.Credentials.from_service_account_info(service_account_info)
client = vision.ImageAnnotatorClient(credentials=credentials)


# === DÉFINITION DES MOTS CLÉS AVEC VARIANTES D'ACCENTS ===
MOTS_DEBIT = {
    "échéance": ["echeance", "echéance", "écheance", "echéance"],
    "indemnités de retard": ["indemnites de retard", "indemnités de retard", "indemnites de retard"],
    "prélèvement impayé": ["prelevement impaye", "prélevement impayé", "prelevement impayé", "prélevement impaye"],
    "indemnité report": ["indemnite report", "indemnité report"],
    "déchéance du terme": ["decheance du terme", "décheance du terme", "dechéance du terme"],
    "indemnité de transmission": ["indemnite de transmission", "indemnité de transmission"]
}

MOTS_CREDIT = {
    "prélèvement banque": ["prelevement banque", "prélevement banque"],
    "prélèvement mso": ["prelevement mso", "prélevement mso"],
    "annulation de retard": ["annulation de retard"],
    "versement cb": ["versement cb"],
    "annulation ird": ["annulation ird"],
    "cheque": ["cheque", "chèque"],
    "annulation indemnités retard": ["annulation indemnites retard", "annulation indemnités retard"]
}

# === FONCTIONS EXISTANTES ===
def pdf_to_images(pdf_bytes):
    return convert_from_bytes(pdf_bytes)

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

# === NOUVELLE FONCTION POUR EXTRAIRE MONTANTS AVEC VARIANTES ===
def extraire_montant_apres_mot(ligne_text, mot_principal):
    # Récupère toutes les variantes du mot
    variantes = [mot_principal] + MOTS_DEBIT.get(mot_principal, MOTS_CREDIT.get(mot_principal, []))
    
    for variante in variantes:
        # Pattern pour trouver le mot suivi directement du montant
        pattern = re.compile(
            r"(" + re.escape(variante) + r")[\s-]*(\d+[\.,]\d{2})\b",
            re.IGNORECASE
        )
        match = pattern.search(ligne_text)
        if match:
            montant_str = match.group(2).replace(',', '.')
            try:
                return float(montant_str), match.group(1)  # Retourne le montant et le mot trouvé
            except ValueError:
                continue
    return None, None

# === FONCTION POUR SURlIGNER LE TEXTE ===
def surligner_texte(ligne_text, mot_trouve, montant):
    # Style CSS pour le surlignage
    style_mot = "background-color: #FFF59D; padding: 2px; border-radius: 3px;"  # Jaune pour le mot-clé
    style_montant = "background-color: #C8E6C9; padding: 2px; border-radius: 3px;"  # Vert pour le montant
    
    # Remplace le mot trouvé
    texte_surligne = ligne_text.replace(
        mot_trouve,
        f"<span style='{style_mot}'>{mot_trouve}</span>"
    )
    
    # Remplace le montant (format XX.XX ou XX,XX)
    montant_str = f"{montant:.2f}".replace('.', '[\.,]')
    pattern_montant = re.compile(r"(\d+[\.,]\d{2})")
    montant_trouve = pattern_montant.search(ligne_text)
    
    if montant_trouve:
        texte_surligne = texte_surligne.replace(
            montant_trouve.group(1),
            f"<span style='{style_montant}'>{montant_trouve.group(1)}</span>"
        )
    
    return texte_surligne

# === INTERFACE STREAMLIT ===
st.set_page_config(page_title="OCR PDF multi-pages", layout="wide")

# === AFFICHAGE DU LOGO ===
if os.path.exists("logo.png"):
    logo_img = Image.open("logo.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo_img)
else:
    st.warning("Logo non trouvé. Assure-toi que 'logo.png' est présent dans le même dossier que ce script.")


st.title("📄 OCR extraction des montants")

# Sélection des mots-clés
st.subheader("🔍 Sélectionnez les mots-clés à rechercher")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Famille DÉBIT**")
    debits_selectionnes = []
    for mot in MOTS_DEBIT:
        if st.checkbox(f"{mot}", key=f"debit_{mot}"):
            debits_selectionnes.append(mot)

with col2:
    st.markdown("**Famille CRÉDIT**")
    credits_selectionnes = []
    for mot in MOTS_CREDIT:
        if st.checkbox(f"{mot}", key=f"credit_{mot}"):
            credits_selectionnes.append(mot)

# Upload du fichier
uploaded_file = st.file_uploader("Dépose ton fichier PDF scanné ici", type=["pdf"])

if uploaded_file and (debits_selectionnes or credits_selectionnes):
    pdf_bytes = uploaded_file.read()
    try:
        images = pdf_to_images(pdf_bytes)
        st.success(f"✅ {len(images)} page(s) PDF analysée(s).")

        line_counter = 0
        total_debit = 0.0
        total_credit = 0.0

        for i, pil_img in enumerate(images):
            pil_img = pil_img.convert("RGB")
            with st.spinner(f"Analyse OCR de la page {i+1}..."):
                words = vision_ocr_detect_text(pil_img)

            if not words:
                continue

            lines = group_words_by_lines(words, y_tolerance=10)

            st.markdown(f"**Page {i+1} - Texte reconnu :**")
            for idx, line in enumerate(lines):
                line_text = " ".join([w['text'] for w in line['words']])
                ligne_num = line_counter + idx + 1
                
                # Vérification des mots-clés
                montant_trouve = None
                mot_trouve = None
                type_montant = None
                
                # D'abord les débits
                for mot in debits_selectionnes:
                    montant, mot_var = extraire_montant_apres_mot(line_text, mot)
                    if montant:
                        montant_trouve = montant
                        mot_trouve = mot_var
                        type_montant = "DÉBIT"
                        total_debit += montant
                        break
                
                # Puis les crédits si pas trouvé en débit
                if not montant_trouve:
                    for mot in credits_selectionnes:
                        montant, mot_var = extraire_montant_apres_mot(line_text, mot)
                        if montant:
                            montant_trouve = montant
                            mot_trouve = mot_var
                            type_montant = "CRÉDIT"
                            total_credit += montant
                            break
                
                # Affichage avec surlignage si montant trouvé
                if montant_trouve:
                    texte_surligne = surligner_texte(line_text, mot_trouve, montant_trouve)
                    st.markdown(
                        f"L{ligne_num} ({type_montant}): {texte_surligne} → "
                        f"<span style='color: {"red" if type_montant == "DÉBIT" else "green"};'>"
                        f"{montant_trouve:.2f} €</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.write(f"L{ligne_num}: {line_text}")

            line_counter += len(lines)

        # Affichage des totaux
        solde_final = total_credit - total_debit
        
        st.subheader("💰 Totaux")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style='background-color:#e6ffe6;padding:15px;border-radius:10px;'>
                <h3 style='color:#007700;'>✅ Total CRÉDIT</h3>
                <h2 style='color:#007700;'>+ {total_credit:.2f} €</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='background-color:#ffe6e6;padding:15px;border-radius:10px;'>
                <h3 style='color:#990000;'>❌ Total DÉBIT</h3>
                <h2 style='color:#990000;'>- {total_debit:.2f} €</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style='background-color:#e6f0ff;padding:15px;border-radius:10px;'>
                <h3 style='color:#003366;'>💰 SOLDE FINAL</h3>
                <h2 style='color:#003366;'>{solde_final:+.2f} €</h2>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Erreur: {e}")
elif uploaded_file:
    st.warning("Veuillez sélectionner au moins un mot-clé à rechercher")
