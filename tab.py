"""
@author: lenapatarin
"""


# === IMPORTS ===
import streamlit as st  # Pour créer une interface web interactive
from pdf2image import convert_from_bytes  # Pour convertir un fichier PDF en images (par page)
from google.oauth2 import service_account  # Pour utiliser une clé API Google Vision de façon sécurisée
from google.cloud import vision  # Bibliothèque Google Cloud Vision pour faire de l'OCR
import io  # Pour manipuler des fichiers en mémoire
from PIL import Image, ImageDraw, ImageFont  # Pour afficher et dessiner sur les images


# === INITIALISATION DU CLIENT GOOGLE VISION ===
import json
from google.oauth2 import service_account
from google.cloud import vision
import streamlit as st

# Chargement des credentials à partir du secret JSON
service_account_info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# Création du client Google Vision
client = vision.ImageAnnotatorClient(credentials=credentials)


# === CONVERSION DU PDF EN IMAGES ===
def pdf_to_images(pdf_bytes):
    return convert_from_bytes(pdf_bytes)  # Convertit chaque page du PDF en une image PIL


# === APPEL À L'OCR DE GOOGLE VISION POUR UNE IMAGE ===
def vision_ocr_detect_text(image_pil):
    img_byte_arr = io.BytesIO()  # Création d'un tampon mémoire pour l’image
    image_pil.save(img_byte_arr, format='PNG')  # Sauvegarde de l'image dans ce tampon
    content = img_byte_arr.getvalue()  # Récupération des données binaires

    image = vision.Image(content=content)  # Préparation de l'image pour Google Vision
    response = client.text_detection(image=image)  # Appel à l’OCR

    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")  # Gestion des erreurs éventuelles

    annotations = response.text_annotations  # Résultats de l’OCR
    if not annotations:
        return []  # Aucun texte détecté

    words = []
    for ann in annotations[1:]:  # On saute le 1er élément (texte global)
        vertices = ann.bounding_poly.vertices  # Coins de la boîte entourant le mot
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))  # Boîte englobante
        words.append({"text": ann.description, "bbox": bbox})  # On stocke le mot et sa position

    return words  # Liste des mots détectés avec position


# === REGROUPER LES MOTS EN LIGNES BASÉ SUR LEUR POSITION Y ===
def group_words_by_lines(words, y_tolerance=10):
    lines = []
    words_sorted = sorted(words, key=lambda w: w['bbox'][1])  # Trie les mots du haut vers le bas

    for w in words_sorted:
        x_min, y_min, x_max, y_max = w['bbox']
        mid_y = (y_min + y_max) / 2  # Centre vertical du mot
        placed = False
        for line in lines:
            line_y = line['y_mean']
            if abs(mid_y - line_y) <= y_tolerance:  # Si le mot est proche verticalement
                line['words'].append(w)
                ys = [ (ww['bbox'][1] + ww['bbox'][3]) / 2 for ww in line['words'] ]
                line['y_mean'] = sum(ys) / len(ys)  # Recalcule la moyenne Y de la ligne
                placed = True
                break
        if not placed:
            lines.append({'y_mean': mid_y, 'words': [w]})  # Nouvelle ligne

    for line in lines:
        line['words'] = sorted(line['words'], key=lambda w: w['bbox'][0])  # Trie gauche → droite

    return lines


# === DESSINER LES LIGNES ET LEUR NUMÉRO SUR L'IMAGE ===
def draw_lines_on_image(image_pil, lines, line_number_offset=0):
    draw = ImageDraw.Draw(image_pil)  # Préparation pour dessiner
    font = ImageFont.load_default()  # Police basique

    for idx, line in enumerate(lines):  # On parcourt chaque ligne détectée, avec son index (idx)
        words = line['words'] # On récupère la liste des mots appartenant à cette ligne
        
        # On cherche les coordonnées extrêmes de tous les mots pour délimiter la ligne entière :
        x_min = min(w['bbox'][0] for w in words)   # Le X le plus à gauche (bord gauche de la ligne)
        y_min = min(w['bbox'][1] for w in words)   # Le Y le plus haut (bord haut de la ligne)
        x_max = max(w['bbox'][2] for w in words)   # Le X le plus à droite (bord droit de la ligne)
        y_max = max(w['bbox'][3] for w in words)   # Le Y le plus bas (bord bas de la ligne) 

        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)  # Encadrement de la ligne
        draw.text((x_min, y_min - 10), f"L{line_number_offset + idx + 1}", fill="red", font=font)  # Numéro ligne

    return image_pil #ça retourne le cadre délimité avec les caractères à l


# === INTERFACE STREAMLIT ===
st.set_page_config(page_title="OCR PDF multi-pages", layout="wide")  # Mise en page large

# === AFFICHAGE DU LOGO EN QUALITÉ MAXIMALE ===
from PIL import Image
import os

logo_path = "/Users/lenapatarin/Desktop/huissiers/logo.png"
if os.path.exists(logo_path):
    logo_img = Image.open(logo_path)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo_img)
else:
    st.warning("Logo introuvable à l'emplacement spécifié.")

st.title("📄 OCR et regroupement des lignes sur toutes les pages PDF")  # Titre principal


uploaded_file = st.file_uploader("Dépose ton fichier PDF scanné ici", type=["pdf"])  # Dépose de PDF"

if uploaded_file:
    pdf_bytes = uploaded_file.read()  # Lit le fichier
    try:
        images = pdf_to_images(pdf_bytes)  # Conversion PDF → images
        st.success(f"✅ {len(images)} page(s) PDF convertie(s) en image(s).")  # Message utilisateur

        line_counter = 0  # Numérotation globale des lignes

        for i, pil_img in enumerate(images):  # Pour chaque page
            pil_img = pil_img.convert("RGB")  # Format RGB
            with st.spinner(f"Analyse OCR Google Vision de la page {i+1}..."):
                words = vision_ocr_detect_text(pil_img)  # OCR Google Vision

            if not words:
                continue  # Page vide → on passe

            lines = group_words_by_lines(words, y_tolerance=10)  # Regroupement en lignes
            annotated_img = draw_lines_on_image(pil_img.copy(), lines, line_number_offset=line_counter)  # Dessin

            st.image(annotated_img, caption="Lignes regroupées et numérotées", use_column_width=True)  # Affichage

            st.markdown("**Texte reconnu par ligne :**")  # Sous-titre
            for idx, line in enumerate(lines):
                line_text = " ".join([w['text'] for w in line['words']])  # Texte complet de la ligne
                st.write(f"L{line_counter + idx + 1}: {line_text}")  # Affichage texte ligne

            line_counter += len(lines)  # Mise à jour compteur global

        if line_counter == 0:
            st.warning("Aucune page contenant du texte détectée dans ce PDF.")  # Si rien trouvé

    except Exception as e:
        st.error(f"Erreur: {e}")  # Gestion d’erreur



