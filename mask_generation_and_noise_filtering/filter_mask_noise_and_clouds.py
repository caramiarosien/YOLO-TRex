import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- DEINE DATEIEN ---
IMG_PATH = "/Users/cara/Desktop/Test_v4/train/images/20211201_frame_0079_GH050036_frames_20211201_Puku.jpg"
MASK_PATH = "/Users/cara/Desktop/Test_v4/mask/20211201_frame_0079_GH050036_frames_20211201_Puku_mask.png"

# --- EINSTELLUNGEN (Physics-Based Cascaded Filtering) ---
# Stufe 1: Geometrie
AREA_NOISE_LIMIT = 3       # Alles < 3px ist sicher Rauschen
AREA_LARGE_KEEP = 60       # Alles > 60px ist sicher kein Wolken-Artefakt (Vogel/Fledermaus nah)

# Stufe 2: Physik (LSNR)
LSNR_THRESHOLD = 1.5       # Kontrast muss 1.5x höher sein als Umgebungs-Rauschen

# Stufe 3: Adaptive Form
AREA_SHAPE_CHECK_MIN = 10  # Erst ab 10px macht Form-Analyse Sinn
SOLIDITY_THRESHOLD = 0.6   # Wolkenfetzen sind oft ausgefranst (< 0.6)

# --- DIE HYBRID-FUNKTION (Cascaded Filtering) ---
def preprocess_mask_cascaded(mask_raw, img_gray):
    """
    3-Stage Cascaded Filtering:
    1. Geometrie (Noise weg, Große behalten)
    2. Physik (LSNR check für den Rest)
    3. Adaptive Form (Solidität nur für mittelgroße Objekte)
    """
    
    # Vorverarbeitung: Salt Noise Removal (ganz feines Rauschen weg)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_pre = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel_open)

    # Analyse der Komponenten
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_pre, connectivity=8)
    
    # Masken vorbereiten
    final_mask = np.zeros_like(mask_raw)         
    rejected_mask = np.zeros_like(mask_raw) # Zum Debuggen (rot anzeigen)
    
    # Statistik-Zähler
    stats_dict = {
        "deleted_noise": 0,
        "kept_large_blind": 0,
        "rejected_lsnr": 0,
        "rejected_solidity": 0,
        "kept_small_lsnr_only": 0, # <10px, nur LSNR erfolgreich
        "kept_medium_full_check": 0 # >10px, LSNR + Form erfolgreich
    }

    # Loop über alle Blobs (start bei 1, da 0 Hintergrund ist)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # ---------------------------------------------------------
        # STUFE 1: GROB-FILTER (GEOMETRIE)
        # ---------------------------------------------------------
        
        # Regel 1.1: Zu klein -> Weg
        if area < AREA_NOISE_LIMIT:
            stats_dict["deleted_noise"] += 1
            continue
            
        # Regel 1.2: Sehr groß -> Sofort behalten (Auto-Accept)
        if area > AREA_LARGE_KEEP:
            final_mask[labels == i] = 255
            stats_dict["kept_large_blind"] += 1
            continue

        # Alles was hier ankommt ist "Mittelklasse" (3px - 60px)
        # Hier ist die Verwechslungsgefahr mit Wolken am größten.
        # -> Wir brauchen Physik!
        
        # Daten vorbereiten
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # ROI Maske (binär nur dieses Objekt)
        roi_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
        
        # ---------------------------------------------------------
        # STUFE 2: PHYSIK-CHECK (LSNR)
        # ---------------------------------------------------------
        
        # Kontext (Ring) definieren
        pad = 5
        x_pad = max(0, x - pad)
        y_pad = max(0, y - pad)
        w_pad = min(mask_raw.shape[1], x + w + pad * 2) - x_pad
        h_pad = min(mask_raw.shape[0], y + h + pad * 2) - y_pad
        
        roi_img_context = img_gray[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
        roi_mask_full = (labels[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad] == i).astype(np.uint8) * 255
        bg_mask = cv2.bitwise_not(roi_mask_full)
        
        mean_obj, std_obj = cv2.meanStdDev(roi_img_context, mask=roi_mask_full)
        mean_bg, std_bg = cv2.meanStdDev(roi_img_context, mask=bg_mask)
        
        sigma_bg = std_bg[0][0]
        mu_obj = mean_obj[0][0]
        mu_bg = mean_bg[0][0]
        
        # LSNR Berechnung
        if sigma_bg > 0:
            lsnr = abs(mu_obj - mu_bg) / sigma_bg
        else:
            lsnr = 999 
            
        if lsnr < LSNR_THRESHOLD:
            # Wolke! (Wenig Kontrast zum unruhigen Hintergrund)
            stats_dict["rejected_lsnr"] += 1
            rejected_mask[labels == i] = 255
            continue # Nächster Kandidat

        # ---------------------------------------------------------
        # STUFE 3: ADAPTIVE FORM-ANALYSE
        # ---------------------------------------------------------
        
        # Wenn wir hier sind, ist der Kontrast okay.
        # Jetzt prüfen wir die Form, ABER nur wenn das Objekt groß genug ist (>10px).
        # Bei winzigen Punkten (3-9px) ist Form Zufall -> Wir vertrauen dem LSNR.
        
        if area < AREA_SHAPE_CHECK_MIN:
            # Zu klein für Form-Check -> AKZEPTIERT (da LSNR ok war)
            final_mask[labels == i] = 255
            stats_dict["kept_small_lsnr_only"] += 1
            continue
            
        # Ab hier: Objekt ist > 10px. Wir können Solidität prüfen.
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        is_frayed = False
        if contours:
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity < SOLIDITY_THRESHOLD:
                    is_frayed = True
        
        if is_frayed:
            # Ausgefranster Wolkenfetzen
            stats_dict["rejected_solidity"] += 1
            rejected_mask[labels == i] = 255
            continue
            
        # Wenn wir hier ankommen: Groß genug, LSNR gut, Form solide -> AKZEPTIERT
        final_mask[labels == i] = 255
        stats_dict["kept_medium_full_check"] += 1

    # 4. Finalisierung: Adaptive Dilation
    # Wir dilatieren das Ergebnis leicht, damit YOLO auch die Ränder lernt (wie empfohlen)
    if np.any(final_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)

    print(f"--- CASCADED FILTER RESULTS ---")
    print(f"🗑️  Stage 1 noise (<3px): {stats_dict['deleted_noise']}")
    print(f"🦅 Stage 1 large (>60px): {stats_dict['kept_large_blind']}")
    print(f"☁️  Stage 2 LSNR rejected: {stats_dict['rejected_lsnr']}")
    print(f"☁️  Stage 3 Solidity rejected: {stats_dict['rejected_solidity']}")
    print(f"✅ Kept (Small, LSNR only): {stats_dict['kept_small_lsnr_only']}")
    print(f"✅ Kept (Medium, Full Check): {stats_dict['kept_medium_full_check']}")
    
    return final_mask, rejected_mask

# --- HAUPTPROGRAMM ---
def main():
    if not os.path.exists(IMG_PATH) or not os.path.exists(MASK_PATH):
        print("❌ Pfade prüfen! Datei nicht gefunden.")
        return

    # 1. Laden
    img = cv2.imread(IMG_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Für Matplotlib
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Für Analyse
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

    # individuals count BEFORE
    contours_original, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count_original = len(contours_original)

    # 2. Verarbeiten
    new_mask, rejected_mask = preprocess_mask_cascaded(mask, img_gray)
    
    # individuals count AFTER
    contours_new, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count_new = len(contours_new)

    print(f"\n--- VERGLEICH INDIVIDUEN ---")
    print(f"🦇 Original Maske: {count_original} Individuen")
    print(f"🦇 Optimierte Maske: {count_new} Individuen")
    print(f"➡️ Differenz: {count_original - count_new} entfernt\n")

    # --- SPEICHERN ---
    filename = os.path.basename(MASK_PATH)
    output_path = MASK_PATH.replace('.png', '_cleaned.png')
    cv2.imwrite(output_path, new_mask)
    print(f"💾 Optimierte Maske gespeichert unter:\n   {output_path}\n")

    # 3. Visualisierung
    
    contours_acc, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_rej, _ = cv2.findContours(rejected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_viz = img_rgb.copy()
    cv2.drawContours(img_viz, contours_rej, -1, (255, 0, 0), 2) # ROT = REJECTED
    cv2.drawContours(img_viz, contours_acc, -1, (0, 255, 0), 2) # GRÜN = ACCEPTED

    # 4. Plotting
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.title(f"Original Mask ({count_original} items)")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Physics Filter Result\nGreen=Bat ({count_new}), Red=Cloud/Noise")
    plt.imshow(img_viz)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()