import numpy as np
import cv2
import matplotlib.pyplot as plt

def anisotropic_diffusion(img, iterations=20, kappa=50, gamma=0.1):
    img = img.astype(np.float64)
    imgout = img.copy()
    
    for _ in range(iterations):
        # Gradienten in 4 Richtungen
        deltaN = np.roll(imgout, 1, axis=0) - imgout
        deltaS = np.roll(imgout, -1, axis=0) - imgout
        deltaE = np.roll(imgout, -1, axis=1) - imgout
        deltaW = np.roll(imgout, 1, axis=1) - imgout
        
        # Diffusionskoeffizienten (richtungsabhängig!)
        cN = np.exp(-(deltaN/kappa)**2)
        cS = np.exp(-(deltaS/kappa)**2)
        cE = np.exp(-(deltaE/kappa)**2)
        cW = np.exp(-(deltaW/kappa)**2)
        
        # Anisotrope Diffusion
        imgout += gamma * (cN*deltaN + cS*deltaS + cE*deltaE + cW*deltaW)
    
    return imgout

# Bild laden
image_path = 'siemens_star.png'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Bild '{image_path}' nicht gefunden. Lege die Datei in denselben Ordner wie dieses Skript oder passe 'image_path' an.")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Filter anwenden
gaussian = cv2.GaussianBlur(img_gray, (15, 15), 0)
anisotropic = anisotropic_diffusion(img_gray, iterations=25, kappa=50, gamma=0.15)

# Kantendetektion zum Vergleich
edges_original = cv2.Canny(img_gray, 50, 150)
edges_gaussian = cv2.Canny(gaussian.astype(np.uint8), 50, 150)
edges_anisotropic = cv2.Canny(anisotropic.astype(np.uint8), 50, 150)

# Visualisierung
fig = plt.figure(figsize=(18, 10))

# Obere Reihe: Geglättete Bilder
ax1 = plt.subplot(2, 3, 1)
ax1.imshow(img_gray, cmap='gray')
ax1.set_title('Original', fontsize=14, fontweight='bold')
ax1.axis('off')

ax2 = plt.subplot(2, 3, 2)
ax2.imshow(gaussian, cmap='gray')
ax2.set_title('ISOTROPISCH', fontsize=14, fontweight='bold', color='red')
ax2.axis('off')

ax3 = plt.subplot(2, 3, 3)
ax3.imshow(anisotropic, cmap='gray')
ax3.set_title('ANISOTROPISCH', fontsize=14, fontweight='bold', color='green')
ax3.axis('off')

# Untere Reihe: Kantendetektion
ax4 = plt.subplot(2, 3, 4)
ax4.imshow(edges_original, cmap='gray')
ax4.set_title('Kanten: Original', fontsize=12)
ax4.axis('off')

ax5 = plt.subplot(2, 3, 5)
ax5.imshow(edges_gaussian, cmap='gray')
ax5.set_title('Kanten: Nach Gaussian\n(Kanten verschwommen)', fontsize=12, color='red')
ax5.axis('off')

ax6 = plt.subplot(2, 3, 6)
ax6.imshow(edges_anisotropic, cmap='gray')
ax6.set_title('Kanten: Nach Anisotropisch\n(Kanten erhalten!)', fontsize=12, color='green')
ax6.axis('off')

plt.suptitle('DIREKTER VERGLEICH: Isotropisch vs. Anisotropisch', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('direct_comparison.png', dpi=300, bbox_inches='tight')
plt.show()