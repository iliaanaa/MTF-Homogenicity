import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter
import sys


# --- ΦΟΡΤΩΣΗ ΕΙΚΟΝΩΝ ---
try:
    phantom_image_color = io.imread('phantom_C9FF1H.gif')
    homogeneity_image_color = io.imread('homogeneity_image_C9FF1A.gif')
    
    # Μετατροπή σε grayscale
    if phantom_image_color.ndim == 3:
        phantom_image = color.rgb2gray(phantom_image_color)
    else:
        phantom_image = phantom_image_color.astype(float)
        
    if homogeneity_image_color.ndim == 3:
        homogeneity_image = color.rgb2gray(homogeneity_image_color)
    else:
        homogeneity_image = homogeneity_image_color.astype(float)

    # Κανονικοποίηση σε 0-255 (uint8)
    phantom_image = (phantom_image * 255).astype(np.uint8)
    homogeneity_image = (homogeneity_image * 255).astype(np.uint8)

except FileNotFoundError:
    print("ΣΦΑΛΜΑ: Δεν βρέθηκαν τα αρχεία εικόνας (.gif).")
    sys.exit()


#  ΕΝΟΤΗΤΑ Α: Μέτρηση MTF για ΚΑΘΕ Τεταρτημόριο ---
print("### Α. Μέτρηση MTF γ-κάμερας (Ανάλυση 4 Τεταρτημορίων) ###")

def calculate_average_modulation(profile: np.ndarray):
    """ Υπολογίζει τη μέση Διαμόρφωση (M), P_max, P_min και επιστρέφει indices. """
    # Εύρεση κορυφών (Λευκές γραμμές)
    peaks_indices, _ = find_peaks(profile, distance=1, prominence=15) # Prominence 15 για ευαισθησία
    p_max_values = profile[peaks_indices]
    
    # Εύρεση κοιλιών (Μαύρες γραμμές)
    troughs_indices, _ = find_peaks(-profile, distance=1, prominence=15)
    p_min_values = profile[troughs_indices]
    
    if len(p_max_values) < 2 or len(p_min_values) < 2:
        return 0.0, 0.0, 0.0, peaks_indices, troughs_indices
    
    P_max_avg = np.mean(p_max_values)
    P_min_avg = np.mean(p_min_values)
    
    if (P_max_avg + P_min_avg) != 0:
        modulation = (P_max_avg - P_min_avg) / (P_max_avg + P_min_avg)
    else:
        modulation = 0.0
        
    return modulation, P_max_avg, P_min_avg, peaks_indices, troughs_indices

#Προετοιμασία για τα 4 Τεταρτημόρια
height, width = phantom_image.shape
mid_h, mid_w = height // 2, width // 2

# Λίστα με τα 4 τεταρτημόρια: (Όνομα, Y_start, Y_end, X_start, X_end)
# Η σειρά συνήθως είναι: Πάνω Αριστερά, Πάνω Δεξιά, Κάτω Αριστερά, Κάτω Δεξιά.
# Οι συχνότητες (7,6,5,4) πρέπει να αντιστοιχηθούν ανάλογα με τη δομή του phantom σας.
# Εδώ βάζουμε γενικούς τίτλους θέσης.
quadrants = [
    ("Πάνω Αριστερά (Q1)", 0, mid_h, 0, mid_w),
    ("Πάνω Δεξιά (Q2)", 0, mid_h, mid_w, width),
    ("Κάτω Αριστερά (Q3 - 5 lp/mm)", mid_h, height, 0, mid_w),
    ("Κάτω Δεξιά (Q4)", mid_h, height, mid_w, width)
]

plt.figure(figsize=(15, 10))
plt.suptitle("Ανάλυση Προφίλ MTF ανά Τεταρτημόριο")

for i, (name, y1, y2, x1, x2) in enumerate(quadrants):
    # 1. Αποκοπή του τεταρτημορίου
    roi = phantom_image[y1:y2, x1:x2]
    h_roi, w_roi = roi.shape
    
    # 2. Αυτόματη Ανίχνευση Κατεύθυνσης Γραμμών
    # Παίρνουμε το μεσαίο οριζόντιο και κάθετο προφίλ του ROI
    center_row = roi[h_roi//2, :]
    center_col = roi[:, w_roi//2]
    
    # Υπολογίζουμε την τυπική απόκλιση (std). Όπου υπάρχει μεγαλύτερη διακύμανση, εκεί "κόβουμε" τις γραμμές.
    # Αν std(row) > std(col) -> Οι γραμμές είναι Κάθετες -> Θέλουμε Οριζόντιο προφίλ.
    # Αν std(col) > std(row) -> Οι γραμμές είναι Οριζόντιες -> Θέλουμε Κάθετο προφίλ.
    if np.std(center_row) > np.std(center_col):
        profile = center_row
        direction = "Οριζόντιο"
        color_line = 'g-' # Πράσινο για οριζόντιο
    else:
        profile = center_col
        direction = "Κάθετο"
        color_line = 'b-' # Μπλε για κάθετο
        
    # 3. Υπολογισμός MTF
    mod, pmax, pmin, p_idx, t_idx = calculate_average_modulation(profile)
    
    # 4. Εκτύπωση αποτελεσμάτων
    print(f"\n--- {name} ---")
    print(f"   Κατεύθυνση Προφίλ: {direction}")
    print(f"   Μέση Διαμόρφωση (MTF): {mod:.4f}")
    print(f"   P_max: {pmax:.2f}, P_min: {pmin:.2f}")

    # 5. Σχεδίαση στο γράφημα (Subplot)
    plt.subplot(2, 2, i+1)
    plt.plot(profile, color='black', alpha=0.6, label='Προφίλ')
    plt.plot(p_idx, profile[p_idx], 'ro', markersize=4, label='Pmax')
    plt.plot(t_idx, profile[t_idx], 'bo', markersize=4, label='Pmin')
    plt.title(f"{name}\nMTF: {mod:.4f}")
    plt.grid(True, alpha=0.3)
    if i == 0: plt.legend()

plt.tight_layout()
plt.show()
print("-" * 50)



# EΝΟΤΗΤΑ Β: Μέτρηση της Ανομοιογένειας Πεδίου γ-κάμερας ---
print("### Β. Μέτρηση της ανομοιογένειας πεδίου γ-κάμερας (Πλήρης Ανάλυση) ###")

def calculate_field_nonuniformity(image: np.ndarray, kernel_size: int = 13):
    
    # 1. Φίλτρο μέσου όρου (Smoothing)
    print(f"Εφαρμογή φίλτρου μέσου όρου {kernel_size}x{kernel_size}...")
    A_p = uniform_filter(image, size=kernel_size)
    
    # 2. Υπολογισμός A_min και A_max (Εύρος)
    A_max = np.max(A_p)
    A_min = np.min(A_p)
    
    max_coords = np.unravel_index(np.argmax(A_p), A_p.shape)
    min_coords = np.unravel_index(np.argmin(A_p), A_p.shape)
    
    # 3. Υπολογισμός Ανομοιογένειας Εύρους
    if A_max != 0:
        range_non_uniformity = (A_max - A_min) / A_max
    else:
        range_non_uniformity = 0.0

    # 4. Υπολογισμός Μέσης Σχετικής Ανομοιογένειας
    I_bar = np.mean(image)
    
    if I_bar != 0:
        relative_local_non_uniformity = np.abs(A_p - I_bar) / I_bar
    else:
        relative_local_non_uniformity = np.zeros_like(A_p)
        
    mean_relative_non_uniformity = np.mean(relative_local_non_uniformity)
    
    # Εκτύπωση
    print("\nΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΟΜΟΙΟΓΕΝΕΙΑΣ ΕΥΡΟΥΣ")
    print(f"   Μέγιστη (A_max): {A_max:.2f} @ {max_coords[::-1]}")
    print(f"   Ελάχιστη (A_min): {A_min:.2f} @ {min_coords[::-1]}")
    print(f"   Ανομοιογένεια Εύρους: **{range_non_uniformity:.4f}**")
    
    print("\nΑΠΟΤΕΛΕΣΜΑΤΑ ΜΕΣΗΣ ΣΧΕΤΙΚΗΣ ΑΝΟΜΟΙΟΓΕΝΕΙΑΣ")
    print(f"   Γενικός Μέσος Όρος (I_bar): {I_bar:.2f}")
    print(f"   Μέση Σχετική Ανομοιογένεια: **{mean_relative_non_uniformity:.4f}**")

    sys.stdout.flush()

    #Οπτικοποίηση
    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap='gray')
    plt.colorbar(label='Ένταση I')
    plt.title(f'Εικόνα I Ομογενούς Phantom & Σημεία A_max, A_min')
    plt.plot(max_coords[1], max_coords[0], 'rx', markersize=15, linewidth=2, label=f'Amax ({A_max:.2f})')
    plt.plot(min_coords[1], min_coords[0], 'b+', markersize=15, linewidth=2, label=f'Amin ({A_min:.2f})')
    plt.legend()
    plt.show()

# Εκτέλεση Ενότητας Β
calculate_field_nonuniformity(homogeneity_image, kernel_size=13)
print("-" * 50)