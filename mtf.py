import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter
import sys

#ΦΟΡΤΩΣΗ ΕΙΚΟΝΩΝ
# 'phantom_C9FF1H.gif' -> Εικόνα του quadrant phantom (Ενότητα Α)
# 'homogeneity_image_C9FF1A.gif' -> Εικόνα ομογενούς phantom (Ενότητα Β)

try:
    # Οι εικόνες φορτώνονται ως έγχρωμες (RGB) ή ασπρόμαυρες.
    # Για την ανάλυση, τις μετατρέπουμε σε grayscale (ασπρόμαυρη κλίμακα).
    phantom_image_color = io.imread('phantom_C9FF1H.gif')
    homogeneity_image_color = io.imread('homogeneity_image_C9FF1A.gif')
    
    # Μετατροπή σε grayscale (απαιτείται για αριθμητικές πράξεις). 
    # Η io.imread ενδέχεται να φορτώσει την εικόνα ως 3D array ακόμα και αν είναι grayscale.
    if phantom_image_color.ndim == 3:
        phantom_image = color.rgb2gray(phantom_image_color)
    else:
        phantom_image = phantom_image_color.astype(float)
        
    if homogeneity_image_color.ndim == 3:
        homogeneity_image = color.rgb2gray(homogeneity_image_color)
    else:
        homogeneity_image = homogeneity_image_color.astype(float)
        

    # Μετατροπή των τιμών pixel στην κλίμακα 0-255 για καλύτερη προσομοίωση της έντασης
    # (αν και η ανάλυση λειτουργεί και με τιμές 0-1)
    phantom_image = (phantom_image * 255).astype(np.uint8)
    homogeneity_image = (homogeneity_image * 255).astype(np.uint8)

except FileNotFoundError:
    print("ΣΦΑΛΜΑ: Βεβαιωθείτε ότι τα αρχεία εικόνας είναι στον ίδιο φάκελο με τον κώδικα.")
    exit()


#ΕΝΟΤΗΤΑ Α: Μέτρηση MTF (Modulation Transfer Function)
print("Α. Μέτρηση MTF γ-κάμερας")
def calculate_average_modulation(profile: np.ndarray) -> tuple:
    """
    Υπολογίζει τη μέση Διαμόρφωση (M) από ένα 1D προφίλ έντασης.
    Χρησιμοποιείται ο τύπος: M = (P_max - P_min) / (P_max + P_min).
    
    Returns:
        tuple: (μέση διαμόρφωση, P_max_avg, P_min_avg)
        tuple: συλλογή αντικειμένων
    """

    
    # 1. Εύρεση όλων των τοπικών μεγίστων (P_max)
    # Προσαρμογή παραμέτρων 'distance' και 'prominence' για να εντοπιστούν οι ραβδώσεις
    #profile	Το σήμα εισόδου όπου αναζητούνται οι κορυφές Pmax

    #distance    Η ελάχιστη οριζόντια απόσταση (σε pixels) που πρέπει να υπάρχει μεταξύ δύο διαδοχικών κορυφών.	1	Η τιμή 1 απενεργοποιεί 
    # ουσιαστικά το κριτήριο απόστασης. Κάθε σημείο μπορεί να είναι κορυφή, εφόσον ικανοποιεί το κριτήριο της prominence (προεξοχής)

    #prominence	Η ελάχιστη κάθετη διαφορά (σε μονάδες έντασης) μεταξύ της κορυφής και του χαμηλότερου σημείου της κοιλάδας πριν και μετά την κορυφή.
    #20	Βοηθάει στον αποκλεισμό του μικρού τυχαίου θορύβου. 
    #Μόνο οι κορυφές που έχουν "πτώση" τουλάχιστον 20 μονάδων έντασης από την κορυφή μέχρι την επόμενη κοιλάδα θεωρούνται πραγματικές ράβδοι.
    peaks_indices, _ = find_peaks(profile, distance=1, prominence=20) #Βρίσκει τις θέσεις των τοπικών μεγίστων (κορυφές) στο προφίλ P.
    #Τα μέγιστα αντιστοιχούν στην ένταση των λευκών γραμμών (Pmax)


    #πίνακας που περιέχει μόνο τις τιμές των τοπικών μεγίστων
    p_max_values = profile[peaks_indices]
    

    # 2. Εύρεση όλων των τοπικών ελαχίστων (P_min)
    #Βρίσκει τις θέσεις των τοπικών ελαχίστων (κοιλίες) αναζητώντας τα μέγιστα του αρνητικού προφίλ.
    # #Τα ελάχιστα αντιστοιχούν στην ένταση των μαύρων γραμμών (Pmin).
    troughs_indices, _ = find_peaks(-profile, distance=1, prominence=20)

    #πίνακας που περιέχει μόνο τις τιμές των τοπικών ελαχίστων
    p_min_values = profile[troughs_indices]
    


#Χρειαζόμαστε τουλάχιστον δύο κορυφές για να υπολογίσουμε έναν αξιόπιστο μέσο όρο Pmax και να επιβεβαιώσουμε ότι 
# το προφίλ περιέχει τουλάχιστον ένα πλήρες ζεύγος γραμμών (μία λευκή και η επόμενη).
    if len(p_max_values) < 2 or len(p_min_values) < 2:
        print("Προσοχή: Δεν εντοπίστηκαν αρκετά ζεύγη γραμμών στο προφίλ. Ελέγξτε τις παραμέτρους.")
        return 0.0, 0.0, 0.0
    
    # 3. Υπολογισμός μέσων τιμών
    #Υπολογίζει τη μέση τιμή όλων των Pax που βρέθηκαν.Βοηθάει στην αντιμετώπιση του θορύβου, δίνοντας μια πιο σταθερή τιμή για τις λευκές ράβδους.
    P_max_avg = np.mean(p_max_values)

    P_min_avg = np.mean(p_min_values)
    
    # 4. Υπολογισμός της Μέσης Διαμόρφωσης
    if (P_max_avg + P_min_avg) != 0:
        #Υπολογίζει τη Μέση Διαμόρφωση (M) 3 εφαρμόζοντας τον τύπο του contrast στο πεδίο έντασης.
        #Αυτό είναι το αριθμητικό αποτέλεσμα της MTF για τη συγκεκριμένη χωρική συχνότητα.
        modulation = (P_max_avg - P_min_avg) / (P_max_avg + P_min_avg)
    else:
        modulation = 0.0
        
    return modulation, P_max_avg, P_min_avg, peaks_indices, troughs_indices

#Εξαγωγή Προφίλ (για το Κάτω Αριστερό Τεταρτημόριο, όπου ειναι η μπλε γραμμή)
# Σύμφωνα με το Σχήμα 1(α), το κάτω αριστερό τεταρτημόριο έχει οριζόντιες ραβδώσεις.
# Το προφίλ πρέπει να είναι ΚΑΘΕΤΟ σε αυτές, δηλαδή κατακόρυφο (όπως η μπλε γραμμή).
height, width = phantom_image.shape
mid_h, mid_w = height // 2, width // 2 #Ορίζει το κέντρο της εικόνας για να στοχεύσει το Κάτω Αριστερό τεταρτημόριο 5lp/mm.

# Επιλέγουμε το κάτω αριστερό τεταρτημόριο (Horizontal Stripes: 5 lp/mm)
# Λαμβάνουμε ένα κατακόρυφο προφίλ στο κέντρο αυτού του τεταρτημορίου.
#Επιλέγει ένα κατακόρυφο τμήμα της εικόνας (μια στήλη) από το κάτω αριστερό τεταρτημόριο.
# Αυτό γίνεται γιατί οι ραβδώσεις εκεί είναι οριζόντιες και το προφίλ πρέπει να είναι κάθετο σε αυτές.
profile_column = phantom_image[mid_h + 10 : height - 10, mid_w // 2]



spatial_frequency = '5 lp/mm' # Σύμφωνα με την εκφώνηση: 7, 6, 5, 4 lp/mm

modulation_result, P_max_avg, P_min_avg, peaks_idx, troughs_idx = calculate_average_modulation(profile_column)

print(f"Ανάλυση για το τεταρτημόριο με χωρική συχνότητα: **{spatial_frequency}**")
print(f"Μέση τιμή P_max: {P_max_avg:.2f}")
print(f"Μέση τιμή P_min: {P_min_avg:.2f}")
print(f"Μέση Διαμόρφωση (Modulation): {modulation_result:.4f}")

# --- Οπτικοποίηση ---
# 1. Εικόνα του phantom με σημειωμένο το προφίλ
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(phantom_image, cmap='gray')
plt.title('Σχήμα 1(α): Quadrant Phantom')
plt.plot([mid_w // 2, mid_w // 2], [mid_h + 10, height - 10], 'b-', linewidth=2, label='Προφίλ P (Κάθετο)')
plt.xlim(0, width)
plt.ylim(height, 0)
plt.legend()

# 2. Το προφίλ P με τα εντοπισμένα P_max και P_min
plt.subplot(1, 2, 2)
plt.plot(profile_column, label='Προφίλ Έντασης P')
# Σχεδιάζουμε τα εντοπισμένα ακρότατα
plt.plot(peaks_idx, profile_column[peaks_idx], 'o', color='red', label='Pmax (Εντοπίστηκε)')
plt.plot(troughs_idx, profile_column[troughs_idx], 'o', color='blue', label='Pmin (Εντοπίστηκε)')
plt.title('Σχήμα 1(β) (Προσομοίωση): Προφίλ Έντασης P')
plt.xlabel('Pixel')
plt.ylabel('Ένταση')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
print("-" * 50)


#ΕΝΟΤΗΤΑ Β: Μέτρηση της Ανομοιογένειας Πεδίου γ-κάμερας ---
print("Β. Μέτρηση της ανομοιογένειας πεδίου γ-κάμερας")
import numpy as np
import matplotlib.pyplot as plt
import sys # Απαραίτητο για το sys.stdout.flush()

# ... (Υποθέτουμε ότι οι image, homogeneity_image, κλπ. έχουν φορτωθεί σωστά στην αρχή του αρχείου)

def calculate_field_nonuniformity(image: np.ndarray, kernel_size: int = 13):
    """
    Υπολογίζει την ανομοιογένεια πεδίου (Non-uniformity) σύμφωνα με την εκφώνηση, 
    προσθέτοντας και τη Μέση Σχετική Ανομοιογένεια.
    """
    
    # 1. Υπολογισμός της μέσης τιμής A(p) σε τετράγωνο πλευράς 13 pixel 
    print(f"Εφαρμογή φίλτρου μέσου όρου {kernel_size}x{kernel_size}...")
    #Υπολογίζει τη μέση τιμή A(p) της εικόνας I σε ένα τετράγωνο πλευράς 13 pixel.
    #Εφαρμόζει ένα φίλτρο που υπολογίζει τον μέσο όρο των τιμών pixel σε μια συγκεκριμένη γειτονιά.
    #Εξομάλυνση Θορύβου: Αφαιρεί τον τυχαίο θόρυβο (shot noise) της γ-κάμερας για να αποκαλυφθεί η υποκείμενη μη ομοιογένεια του ανιχνευτή.
    A_p = uniform_filter(image, size=kernel_size)#Η πλευρά του τετραγωνικού "παραθύρου" (kernel) πάνω στο οποίο υπολογίζεται ο μέσος όρος.
    
    # 2. Υπολογισμός A_min και A_max 
    #Βρίσκει τη μέγιστη και ελάχιστη τιμή στην ομαλοποιημένη εικόνα A(p).Αυτές οι τιμές ορίζουν το εύρος της ανομοιογένειας του πεδίου.
    A_max = np.max(A_p)
    A_min = np.min(A_p)
    
    # 3. Εύρεση των συντεταγμένων των A_min και A_max (πρώτο σημείο εμφάνισης)
    max_coords_flat_idx = np.argmax(A_p) #ελάχιστο pixel στον πίνακα Ap.
    #Βρίσκει τις συντεταγμένες (σειρά, στήλη) του pixel όπου εμφανίζεται το Amax.Εντοπίζει το πιο "θερμό" (hot spot) σημείο στην ομαλοποιημένη εικόνα.
    max_coords = np.unravel_index(max_coords_flat_idx, A_p.shape)
    
    min_coords_flat_idx = np.argmin(A_p)
    min_coords = np.unravel_index(min_coords_flat_idx, A_p.shape) #διδιάστατες συντεταγμένες (σειρά, στήλη) του ελάχιστου pixel.
    
    # 4. Υπολογισμός της Ανομοιογένειας Εύρους
    # Τύπος: (A_max - A_min) / A_max
    if A_max != 0:
        range_non_uniformity = (A_max - A_min) / A_max
    else:
        range_non_uniformity = 0.0

   
    #ΥΠΟΛΟΓΙΣΜΟΣ ΜΕΣΗΣ ΣΧΕΤΙΚΗΣ ΑΝΟΜΟΙΟΓΕΝΕΙΑΣ
    # 5. Υπολογισμός Γενικού Μέσου Όρου (I_bar) της αρχικής εικόνας I
    I_bar = np.mean(image)
    
    # 6. Υπολογισμός της Σχετικής Τοπικής Ανομοιογένειας R(p) για κάθε σημείο: R(p) = |A(p) - I_bar| / I_bar
    if I_bar != 0:
        relative_local_non_uniformity = np.abs(A_p - I_bar) / I_bar
    else:
        relative_local_non_uniformity = np.zeros_like(A_p)
        
    # 7. Υπολογισμός της Μέσης Σχετικής Ανομοιογένειας (R_bar)
    mean_relative_non_uniformity = np.mean(relative_local_non_uniformity)
    
    # Εκτύπωση Αποτελεσμάτων
    print("\nΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΟΜΟΙΟΓΕΝΕΙΑΣ ΕΥΡΟΥΣ")
    print(f"Μέγιστη Ομαλοποιημένη Τιμή (A_{{\max}}): {A_max:.2f} (Συντεταγμένες: {max_coords[::-1]} - [στήλη, σειρά])")
    print(f"Ελάχιστη Ομαλοποιημένη Τιμή (A_{{\min}}): {A_min:.2f} (Συντεταγμένες: {min_coords[::-1]} - [στήλη, σειρά])")
    print(f"Ανομοιογένεια Εύρους: (A_max - A_min) / A_max = **{range_non_uniformity:.4f}**")
    
    print("\nΑΠΟΤΕΛΕΣΜΑΤΑ ΜΕΣΗΣ ΣΧΕΤΙΚΗΣ ΑΝΟΜΟΙΟΓΕΝΕΙΑΣ ---")
    print(f"Γενικός Μέσος Όρος Εικόνας (I_bar): {I_bar:.2f}")
    print(f"Μέση Σχετική Ανομοιογένεια: **{mean_relative_non_uniformity:.4f}**")


    #Αναγκάζει την έξοδο να εμφανιστεί αμέσως στο Terminal
    sys.stdout.flush()
    # --- Οπτικοποίηση (Σχεδιάστε την εικόνα Ι και τα δύο pixels)
    plt.figure(figsize=(10, 6))
    
    # Η αρχική εικόνα I
    plt.imshow(image, cmap='gray')
    plt.colorbar(label='Ένταση I')
    plt.title(f'Εικόνα I Ομογενούς Phantom & Σημεία A_{{\max}}, A_{{\min}}')
    
    # Σχεδιάζουμε τα σημεία A_max και A_min στην αρχική εικόνα I
    # Σημείωση: το imshow χρησιμοποιεί συντεταγμένες (σειρά, στήλη)
    plt.plot(max_coords[1], max_coords[0], 'rx', markersize=15, linewidth=2, label=f'A_{{\max}} ({A_max:.2f})')
    plt.plot(min_coords[1], min_coords[0], 'b+', markersize=15, linewidth=2, label=f'A_{{\min}} ({A_min:.2f})')
    plt.legend()
    plt.show()



# Εκτέλεση της ανάλυσης ανομοιογένειας
calculate_field_nonuniformity(homogeneity_image, kernel_size=13)
print("-" * 50)