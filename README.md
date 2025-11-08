# Computer Vision Praxis - Filter Vergleich

## Anisotrope vs. Isotrope Filter

Dieses Repository enthält eine Präsentation und Code-Beispiele zum Vergleich von anisotropen und isotropen Bildfiltern.

### Ordnerstruktur

```
Präsentation/
└── Anisotropischer Filter Vergleich/
    ├── anisotropic.py              # Anisotrope Filter (Perona-Malik)
    ├── isotropic.py                # Isotrope Filter (Gaussian, Median, etc.)
    ├── comparison.py               # Direkter Vergleich beider Filtertypen
    ├── filter_praesentation.bat    # Ein-Klick Launcher für Präsentation
    ├── siemens_star.png            # Testbild (Siemens-Stern)
    └── *.png                       # Generierte Vergleichsbilder
```

### Schnellstart für Präsentation

1. **Einfache Ausführung:**
   ```
   Doppelklick auf filter_praesentation.bat
   ```
   
2. **Manuelle Ausführung:**
   ```bash
   cd "Präsentation/Anisotropischer Filter Vergleich"
   python comparison.py
   ```

### Was wird demonstriert?

- **Isotrope Filter:** Glätten gleichmäßig in alle Richtungen → Kanten werden unscharf
- **Anisotrope Filter:** Glätten nur parallel zu Kanten → Kanten bleiben erhalten
- **Anwendung:** Entrauschen ohne Kantenverlust

### Ergebnisse

- `direct_comparison.png` - Hauptvergleich für Präsentationen
- `isotropic_comparison.png` - Details verschiedener isotroper Filter  
- `anisotropic_comparison.png` - Details anisotroper Diffusion

### Abhängigkeiten

- Python 3.x
- numpy
- opencv-python
- matplotlib

### Filter-Algorithmen

#### Isotrop (Gaussian Blur)
- Konvolution mit Gaussian-Kernel
- Gleichmäßige Glättung in alle Richtungen
- Schnell, aber Kantenverlust

#### Anisotrop (Perona-Malik Diffusion)
- Richtungsabhängige Diffusion
- Erhaltung von Kanten durch adaptive Diffusionskoeffizienten
- Langsamer, aber kantenerhaltend

---
**Erstellt für Computer Vision Praxis**