import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- EINSTELLUNGEN ---
# HIER DEN KOMPLETTEN PFAD EINTRAGEN:
mask_path = "/Users/cara/Desktop/BA/YOLO/Kasanka_YOLO_Dataset_train_mask/GH030007_frames_frame_0203_mask.png"

# Der Output wird in deinen YOLO Ordner gespeichert
output_txt_path = "/Users/cara/Desktop/BA/YOLO/GH030007_frames_frame_0203_cleaned.txt"

min_area_filter = 10.0  # Filter: Alles unter 10 Pixeln wird gelöscht

# 1. Maske laden
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    print(f"❌ Fehler: Konnte Bild immer noch nicht finden.")
    print(f"   Bitte prüfe, ob diese Datei wirklich existiert:\n   {mask_path}")
else:
    h, w = mask.shape[:2]
    print(f"✅ Maske geladen: {w}x{h} Pixel")

    # === VORHER (Das Problem simulieren) ===
    # Wir nehmen alles > 0 ohne Filter
    _, mask_noisy = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours_noisy, _ = cv2.findContours(mask_noisy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Bild für Vorschau (Rot = Rauschen)
    img_before = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img_before, contours_noisy, -1, (255, 0, 0), 1)


    # === NACHHER (Die Lösung anwenden) ===
    # A. Thresholding: Graue Artefakte entfernen (nur Pixel > 127 gelten)
    _, mask_clean = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # B. Konturen finden
    contours_clean, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_lines = []
    img_after = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2RGB)
    valid_objects = 0
    
    for cnt in contours_clean:
        # C. Area Filter: Zu kleine Objekte ignorieren
        if cv2.contourArea(cnt) < min_area_filter:
            continue
            
        # Optional: Glätten
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(cnt) < 3: continue

        # Bild für Vorschau (Grün = Gut)
        cv2.drawContours(img_after, [cnt], -1, (0, 255, 0), 2)
        valid_objects += 1

        # YOLO Zeile generieren
        cnt_norm = cnt.reshape(-1, 2).astype(np.float32)
        cnt_norm[:, 0] /= w
        cnt_norm[:, 1] /= h
        np.clip(cnt_norm, 0.0, 1.0, out=cnt_norm)
        coords = " ".join(f"{v:.6f}" for v in cnt_norm.flatten())
        yolo_lines.append(f"0 {coords}")

    # === SPEICHERN ===
    with open(output_txt_path, 'w') as f:
        f.write("\n".join(yolo_lines))
    
    print(f"✅ Datei gespeichert!")
    print(f"📂 Speicherort: {output_txt_path}")
    print(f"📊 Statistik: {len(contours_noisy)} Konturen (Vorher) -> {valid_objects} Objekte (Nachher)")

    # === VORSCHAU ===
    # Plottet das Ergebnis direkt
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    axes[0].imshow(img_before)
    axes[0].set_title(f"VORHER: {len(contours_noisy)} Objekte (Rot=Rauschen)", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(img_after)
    axes[1].set_title(f"NACHHER: {valid_objects} Objekte (Grün=Gefiltert)", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()