import cv2
import numpy as np
import os
import glob
from pathlib import Path

# --- KONFIGURATION ---
MASK_DIR = "/Users/cara/Desktop/Test_v4_mask/Old"
OUTPUT_DIR = "/Users/cara/Desktop/Test_v4_mask" # Same or different folder

def preprocess_mask_clean_and_dilate(mask_raw):
    # 1. Hard Thresholding (gegen graue Artefakte)
    _, binary = cv2.threshold(mask_raw, 50, 255, cv2.THRESH_BINARY)
    
    # 2. Analyse der Komponenten
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Masken vorbereiten
    final_mask = np.zeros_like(mask_raw)         
    small_objects_mask = np.zeros_like(mask_raw) 
    
    deleted_count = 0
    pumped_count = 0
    kept_original_count = 0

    # Loop über alle Blobs (start bei 1, da 0 Hintergrund ist)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # --- REGEL 1: Rauschen löschen (< 4px) ---
        if area < 4: 
            deleted_count += 1
            continue 
            
        # --- REGEL 2: Kleine aufpumpen (4-25px) ---
        if area < 25:
            small_objects_mask[labels == i] = 255
            pumped_count += 1
        # --- REGEL 3: Große behalten (>25px) ---
        else:
            final_mask[labels == i] = 255
            kept_original_count += 1
            
    # Dilation anwenden (nur auf die Kleinen)
    if np.any(small_objects_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_small = cv2.dilate(small_objects_mask, kernel, iterations=1)
        
        # Zusammenfügen
        final_mask = cv2.bitwise_or(final_mask, dilated_small)
        
    return final_mask, deleted_count, pumped_count, kept_original_count

def main():
    if not os.path.exists(MASK_DIR):
        print(f"❌ Pfad nicht gefunden: {MASK_DIR}")
        return

    # Alle Masken finden (nur die originalen, nicht bereits bereinigte)
    mask_files = sorted([f for f in glob.glob(os.path.join(MASK_DIR, "*_mask.png")) if "_cleaned" not in f])
    
    print(f"Gefunden: {len(mask_files)} Masken in {MASK_DIR}")
    
    total_deleted = 0
    total_pumped = 0
    
    for mask_path in mask_files:
        filename = os.path.basename(mask_path)
        print(f"Verarbeite: {filename} ...")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  ❌ Konnte nicht gelesen werden!")
            continue

        # Verarbeiten
        new_mask, deleted, pumped, kept = preprocess_mask_clean_and_dilate(mask)
        
        total_deleted += deleted
        total_pumped += pumped
        
        # Speichern
        output_filename = filename.replace('.png', '_cleaned.png')
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, new_mask)
        print(f"  ✅ Gespeichert: {output_filename} (Gelöscht: {deleted}, Gepumpt: {pumped})")

    print(f"\n--- GESAMT ---")
    print(f"Gelöschte Objekte: {total_deleted}")
    print(f"Aufgepumpte Objekte: {total_pumped}")
    print("Fertig!")

if __name__ == "__main__":
    main()