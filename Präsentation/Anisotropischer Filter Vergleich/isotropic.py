import numpy as np
import cv2
import matplotlib.pyplot as plt

# Bild laden
image_path = 'siemens_star.png'
img = cv2.imread(image_path)
if img is None:
	raise FileNotFoundError(f"Bild '{image_path}' nicht gefunden. Lege die Datei in denselben Ordner wie dieses Skript oder passe 'image_path' an.")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Verschiedene isotrope Filter anwenden
# 1. Gaussian Blur (klassischer isotroper Filter)
gaussian_blur = cv2.GaussianBlur(img_rgb, (15, 15), 0)

# 2. Median Filter (auch isotrop)
median_blur = cv2.medianBlur(img_rgb, 15)

# 3. Box Filter (Mean Filter - isotrop)
box_blur = cv2.blur(img_rgb, (15, 15))

# 4. Bilateral Filter (teilweise isotrop, aber kantenerhaltend)
bilateral = cv2.bilateralFilter(img_rgb, 15, 80, 80)

# Visualisierung
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Isotropische Filter', fontsize=16, fontweight='bold')

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(gaussian_blur)
axes[0, 1].set_title('Gaussian Blur', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(median_blur)
axes[0, 2].set_title('Median Filter', fontsize=12)
axes[0, 2].axis('off')

axes[1, 0].imshow(box_blur)
axes[1, 0].set_title('Box Filter', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(bilateral)
axes[1, 1].set_title('Bilateral Filter', fontsize=12)
axes[1, 1].axis('off')

# Differenzbild zum Original zeigen
diff = np.abs(img_rgb.astype(float) - gaussian_blur.astype(float))
axes[1, 2].imshow(diff.astype(np.uint8))
axes[1, 2].set_title('Differenz: Original - Gaussian', fontsize=12)
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('isotropic_comparison.png', dpi=300, bbox_inches='tight')
plt.show()