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
from unidecode import unidecode
from PIL import ImageFont

font_path = "fonts/DejaVuSans.ttf"
font = ImageFont.truetype(font_path, size=20)

# === INITIALISATION DU CLIENT GOOGLE VISION (Streamlit Secrets) ===
service_account_info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
credentials = service_account.Credentials.from_service_account_info(service_account_info)
client = vision.ImageAnnotatorClient(credentials=credentials)


# === FONCTION POUR GÉNÉRER LES VARIANTES D'ACCENTS ===
def generer_variantes(mot):
    """Génère toutes les variantes avec/sans accents et combinaisons"""
    variantes = set()
    
    # Version originale
    variantes.add(mot)
    
    # Version sans aucun accent
    sans_accent = unidecode(mot)
    variantes.add(sans_accent)
    
    # Versions avec combinaisons d'accents partiels
    if any(c in mot for c in "éèêëàâäîïôöùûüç"):
        mots_partiels = []
        for i, c in enumerate(mot):
            if c in "éèêë":
                # Variantes pour chaque e accentué
                mots_partiels.extend([
                    mot[:i] + 'e' + mot[i+1:],
                    mot[:i] + 'é' + mot[i+1:],
                    mot[:i] + 'è' + mot[i+1:],
                ])
            elif c in "àâä":
                # Variantes pour chaque a accentué
                mots_partiels.extend([
                    mot[:i] + 'a' + mot[i+1:],
                    mot[:i] + 'à' + mot[i+1:],
                ])
        
        # Ajoute les combinaisons générées
        for mp in mots_partiels:
            variantes.add(mp)
    
    return list(variantes)

# === DÉFINITION DES MOTS CLÉS AVEC VARIANTES ===
def creer_dictionnaire_mots(mots_base):
    """Crée un dictionnaire avec toutes les variantes de chaque mot"""
    nouveau_dict = {}
    for mot, anciennes_variantes in mots_base.items():
        # Génère toutes les nouvelles variantes
        toutes_variantes = generer_variantes(mot)
        # Ajoute les anciennes variantes si existent
        if anciennes_variantes:
            for v in anciennes_variantes:
                toutes_variantes.extend(generer_variantes(v))
        # Élimine les doublons
        toutes_variantes = list(set(toutes_variantes))
        nouveau_dict[mot] = toutes_variantes
    return nouveau_dict

# Dictionnaires de base
MOTS_CREDIT_BASE = {
    "prélevé sur votre compte bancaire": ["preleve sur votre compte bancaire"],
    "votre prélèvement": ["votre prelevement"],
    "votre règlement par cb": ["votre reglement par cb"],
    "votre règlement par ccp": ["votre reglement par ccp"]
}

MOTS_DEBIT_BASE = {
    "cotis cb prélevée banque FOMO": ["cotis cb prelevee banque FOMO"],
    "solde FMRB précédent": ["solde FMRB precedent"],
    "retour de prélèvement impayé": ["retour de prelevement impaye"],
    "indemnité de retard": ["indemnite de retard"],
    "remise à jour de vos impayés": ["remise a jour de vos impayes"],
    "transfert sur votre carte AURORE": [],
    "transfert différé/crédit": ["transfert differe/credit"],
    "régul d'agios": ["regul d'agios"],
    "remise à jour de vos intérêts": ["remise a jour de vos interets"],
    "votre utilisation": [],
    "arrêté de compte": ["%"],
    "trans. différé precedent/credit": [
        "trans. differe precedent/credit",
        "trans . differe precedent/credit"
    ]
}

MOTS_CREDIT_CLASSIQUE_BASE = {
    "prélèvement banque": ["prelevement banque"],
    "prélèvement mso": ["prelevement mso"],
    "annulation de retard": [],
    "versement cb": [],
    "annulation ird": [],
    "cheque": [],
    "annulation indemnités retard": ["annulation indemnites retard"]
}

MOTS_DEBIT_CLASSIQUE_BASE = {
    "échéance": ["echeance"],
    "indemnités de retard": ["indemnites de retard"],
    "prélèvement impayé": ["prelevement impaye"],
    "indemnité report": ["indemnite report"],
    "déchéance du terme": ["decheance du terme"],
    "indemnité de transmission": ["indemnite de transmission"]
}

# Création des dictionnaires complets avec variantes
MOTS_CREDIT_RENOUVELABLE = creer_dictionnaire_mots(MOTS_CREDIT_BASE)
MOTS_DEBIT_RENOUVELABLE = creer_dictionnaire_mots(MOTS_DEBIT_BASE)
MOTS_CREDIT_CLASSIQUE = creer_dictionnaire_mots(MOTS_CREDIT_CLASSIQUE_BASE)
MOTS_DEBIT_CLASSIQUE = creer_dictionnaire_mots(MOTS_DEBIT_CLASSIQUE_BASE)

# === FONCTIONS OCR ===
def pdf_to_images(pdf_bytes):
    return convert_from_bytes(pdf_bytes)

def group_words_by_lines(words, y_tolerance=10):
    """Regroupe les mots en lignes basées sur leur position Y"""
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
                ys = [(ww['bbox'][1] + ww['bbox'][3]) / 2 for ww in line['words']]
                line['y_mean'] = sum(ys) / len(ys)
                placed = True
                break
        if not placed:
            lines.append({'y_mean': mid_y, 'words': [w]})

    for line in lines:
        line['words'] = sorted(line['words'], key=lambda w: w['bbox'][0])

    return lines

def vision_ocr_detect_text(image_pil):
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    content = img_byte_arr.getvalue()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    
    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")

    if not response.text_annotations:
        return []

    words = []
    for ann in response.text_annotations[1:]:  # Skip the first element (full text)
        vertices = ann.bounding_poly.vertices
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        words.append({"text": ann.description, "bbox": bbox})

    return words

def detecter_type_document(images):
    """Détecte si c'est un prêt classique ou crédit renouvelable"""
    for pil_img in images:
        words = vision_ocr_detect_text(pil_img.convert("RGB"))
        if not words:
            continue
            
        lines = group_words_by_lines(words)
        for line in lines:
            line_text = " ".join([w['text'] for w in line['words']])
            if "affaire" in line_text.lower():
                return "Crédit renouvelable"
    return "Prêt classique"

# === FONCTIONS D'EXTRACTION ===
def extraire_montant_apres_mot(ligne_text, mot_principal, mots_reference):
    # Cas spécial pour l'arrêté de compte
    if mot_principal == "arrêté de compte":
        pattern = re.compile(r"%[\s-]*(\d+[\.,]\d{2})\b")
        match = pattern.search(ligne_text)
        if match:
            montant_str = match.group(1).replace(',', '.')
            try:
                return float(montant_str), "%"
            except ValueError:
                return None, None
    
    # Récupère toutes les variantes du mot
    variantes = [mot_principal] + mots_reference.get(mot_principal, [])
    
    for variante in variantes:
        pattern = re.compile(
            r"(" + re.escape(variante) + r")[\s-]*(\d+[\.,]\d{2})\b",
            re.IGNORECASE
        )
        match = pattern.search(ligne_text)
        if match:
            montant_str = match.group(2).replace(',', '.')
            try:
                return float(montant_str), match.group(1)
            except ValueError:
                continue
    return None, None

def surligner_texte(ligne_text, mot_trouve, montant):
    style_mot = "background-color: #FFF59D; padding: 2px; border-radius: 3px;"
    style_montant = "background-color: #C8E6C9; padding: 2px; border-radius: 3px;"
    
    if mot_trouve == "%":
        texte_surligne = ligne_text.replace("%", f"<span style='{style_mot}'>%</span>")
    else:
        texte_surligne = ligne_text.replace(mot_trouve, f"<span style='{style_mot}'>{mot_trouve}</span>")
    
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
st.set_page_config(page_title="Analyse de documents bancaires", layout="wide")

# === AFFICHAGE DU LOGO ===
if os.path.exists("logo.png"):
    logo_img = Image.open("logo.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo_img)
else:
    st.warning("Logo non trouvé. Assure-toi que 'logo.png' est présent dans le même dossier que ce script.")

st.title("🏦 Analyse de documents bancaires")

# Upload du fichier
uploaded_file = st.file_uploader("Déposez votre document PDF", type=["pdf"])

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    try:
        images = pdf_to_images(pdf_bytes)
        
        # Détection du type de document
        with st.spinner("Analyse du type de document..."):
            type_doc = detecter_type_document(images)
        
        # Affichage clair du type de document dans un cadre visible
        st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:20px;'>
            <h2 style='color:#333; text-align:center; font-weight:bold;'>Nature du document : {type_doc}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Sélection des mots-clés appropriés
        if type_doc == "Crédit renouvelable":
            mots_credit = MOTS_CREDIT_RENOUVELABLE
            mots_debit = MOTS_DEBIT_RENOUVELABLE
        else:
            mots_credit = MOTS_CREDIT_CLASSIQUE
            mots_debit = MOTS_DEBIT_CLASSIQUE
        
        # Sélection des mots-clés
        st.subheader("🔍 Sélectionnez les mots-clés à rechercher")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Famille DÉBIT**")
            debits_selectionnes = []
            for mot in mots_debit:
                if st.checkbox(f"{mot}", key=f"debit_{mot}"):
                    debits_selectionnes.append(mot)
        
        with col2:
            st.markdown("**Famille CRÉDIT**")
            credits_selectionnes = []
            for mot in mots_credit:
                if st.checkbox(f"{mot}", key=f"credit_{mot}"):
                    credits_selectionnes.append(mot)
        
        if not (debits_selectionnes or credits_selectionnes):
            st.warning("Veuillez sélectionner au moins un mot-clé à rechercher")
            st.stop()
        
        # Analyse du document
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
                
                montant_trouve = None
                mot_trouve = None
                type_montant = None
                
                # D'abord les débits
                for mot in debits_selectionnes:
                    montant, mot_var = extraire_montant_apres_mot(line_text, mot, mots_debit)
                    if montant:
                        montant_trouve = montant
                        mot_trouve = mot_var
                        type_montant = "DÉBIT"
                        total_debit += montant
                        break
                
                # Puis les crédits si pas trouvé en débit
                if not montant_trouve:
                    for mot in credits_selectionnes:
                        montant, mot_var = extraire_montant_apres_mot(line_text, mot, mots_credit)
                        if montant:
                            montant_trouve = montant
                            mot_trouve = mot_var
                            type_montant = "CRÉDIT"
                            total_credit += montant
                            break
                
                # Affichage avec surlignage
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
from fpdf import FPDF
import tempfile

# Ajout du bouton pour générer le PDF
if st.button("📄 Générer un rapport PDF"):
    class PDF(FPDF):
        def header(self):
            if os.path.exists("logo.png"):
                self.image("logo.png", x=80, y=10, w=50)
                self.ln(30)
            self.set_font("Arial", "B", 16)
            self.cell(0, 10, "Rapport d'analyse bancaire", ln=True, align="C")
            self.ln(10)

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 12)

    # Sommes finales
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Total CRÉDIT : {total_credit:.2f} €", ln=True)
    pdf.cell(0, 10, f"Total DÉBIT : {total_debit:.2f} €", ln=True)
    pdf.cell(0, 10, f"Solde final : {total_credit - total_debit:.2f} €", ln=True)
    pdf.ln(5)

    # Détails crédits
    if 'détails_crédits' in locals():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Détail des CRÉDITS :", ln=True)
        pdf.set_font("Arial", "", 12)
        for label, montant in détails_crédits:
            pdf.cell(0, 10, f"- {label} : {montant:.2f} €", ln=True)
        pdf.ln(5)

    # Détails débits
    if 'détails_débits' in locals():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Détail des DÉBITS :", ln=True)
        pdf.set_font("Arial", "", 12)
        for label, montant in détails_débits:
            pdf.cell(0, 10, f"- {label} : {montant:.2f} €", ln=True)

    # Sauvegarde et téléchargement temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        with open(tmp_file.name, "rb") as f:
            st.download_button(
                label="📥 Télécharger le rapport PDF",
                data=f,
                file_name="rapport_bancaire.pdf",
                mime="application/pdf"
            )
