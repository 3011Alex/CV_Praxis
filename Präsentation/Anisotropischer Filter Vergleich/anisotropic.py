import numpy as np
import cv2
import matplotlib.pyplot as plt

def anisotropic_diffusion(img, iterations=20, kappa=50, gamma=0.1, option=1):
    img = img.astype(np.float64)
    imgout = img.copy()
    
    for _ in range(iterations):
        # Gradienten in 4 Richtungen berechnen
        deltaN = np.roll(imgout, 1, axis=0) - imgout  # Nord
        deltaS = np.roll(imgout, -1, axis=0) - imgout  # Süd
        deltaE = np.roll(imgout, -1, axis=1) - imgout  # Ost
        deltaW = np.roll(imgout, 1, axis=1) - imgout  # West
        
        # Diffusionskoeffizienten berechnen (richtungsabhängig!)
        if option == 1:
            # Bevorzugt breite Bereiche
            cN = np.exp(-(deltaN/kappa)**2)
            cS = np.exp(-(deltaS/kappa)**2)
            cE = np.exp(-(deltaE/kappa)**2)
            cW = np.exp(-(deltaW/kappa)**2)
        else:
            # Bevorzugt hohen Kontrast
            cN = 1.0 / (1.0 + (deltaN/kappa)**2)
            cS = 1.0 / (1.0 + (deltaS/kappa)**2)
            cE = 1.0 / (1.0 + (deltaE/kappa)**2)
            cW = 1.0 / (1.0 + (deltaW/kappa)**2)
        
        # Anisotrope Diffusion: Glättung nur parallel zu Kanten
        imgout += gamma * (cN*deltaN + cS*deltaS + cE*deltaE + cW*deltaW)
    
    return imgout

# Bild laden
image_path = 'siemens_star.png'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Bild '{image_path}' nicht gefunden.")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Anisotrope Filter anwenden
# 1. Perona-Malik Option 1 (wenige Iterationen)
aniso_light = anisotropic_diffusion(img_gray, iterations=10, kappa=50, gamma=0.15, option=1)

# 2. Perona-Malik Option 1 (mehr Iterationen)
aniso_medium = anisotropic_diffusion(img_gray, iterations=30, kappa=50, gamma=0.15, option=1)

# 3. Perona-Malik Option 2 (hoher Kontrast)
aniso_contrast = anisotropic_diffusion(img_gray, iterations=20, kappa=30, gamma=0.15, option=2)

# 4. Anisotropic für Farbbild (pro Kanal)
aniso_color = np.zeros_like(img_rgb, dtype=np.float64)
for i in range(3):
    aniso_color[:,:,i] = anisotropic_diffusion(img_rgb[:,:,i], iterations=20, kappa=50, gamma=0.15, option=1)

# Zum Vergleich: Gaussian Blur (isotrop)
gaussian_blur = cv2.GaussianBlur(img_gray, (15, 15), 0)

# Visualisierung
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Anisotrope Filter - Glätten richtungsabhängig (Kanten bleiben erhalten)', fontsize=16, fontweight='bold')

axes[0, 0].imshow(img_gray, cmap='gray')
axes[0, 0].set_title('Original (Grayscale)', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(aniso_light, cmap='gray')
axes[0, 1].set_title('Anisotropic (10 iterationen, κ=50)', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(aniso_medium, cmap='gray')
axes[0, 2].set_title('Anisotropic (30 iterationen, κ=50)', fontsize=12)
axes[0, 2].axis('off')

axes[1, 0].imshow(aniso_contrast, cmap='gray')
axes[1, 0].set_title('Anisotropic Option 2 (20 iterationen, κ=30)', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(aniso_color.astype(np.uint8))
axes[1, 1].set_title('Anisotropic (Farbe, 20 iterationen)', fontsize=12)
axes[1, 1].axis('off')

axes[1, 2].imshow(gaussian_blur, cmap='gray')
axes[1, 2].set_title('Vergleich: Gaussian Blur (isotrop)', fontsize=12)
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('anisotropic_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
