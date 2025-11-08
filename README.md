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
    ├── siemens_star.png            # Testbild (Siemens-Stern)
    ├── direct_comparison.png       # Hauptvergleich (generiert)
    ├── isotropic_comparison.png    # Isotrope Filter Details (generiert)
    └── anisotropic_comparison.png  # Anisotrope Filter Details (generiert)
```

### Schnellstart für Präsentation

**Manuelle Ausführung:**
```bash
cd "Präsentation/Anisotropischer Filter Vergleich"
python comparison.py
```

**Alle Skripte einzeln ausführen:**
```bash
python isotropic.py      # Verschiedene isotrope Filter
python anisotropic.py    # Anisotrope Diffusion Details  
python comparison.py     # Direkter Vergleich (empfohlen)
```

### Was wird demonstriert?

- **Isotrope Filter:** Glätten gleichmäßig in alle Richtungen → Kanten werden unscharf
- **Anisotrope Filter:** Glätten nur parallel zu Kanten → Kanten bleiben erhalten
- **Anwendung:** Entrauschen ohne Kantenverlust

### Ergebnisse

Die Skripte generieren folgende Visualisierungen:

- `direct_comparison.png` - **Hauptvergleich für Präsentationen** (empfohlen)
- `isotropic_comparison.png` - Details verschiedener isotroper Filter  
- `anisotropic_comparison.png` - Details anisotroper Diffusion mit verschiedenen Parametern

Alle Bilder werden automatisch gespeichert und können direkt in Präsentationen verwendet werden.

### Installation & Ausführung

**Voraussetzungen:**
```bash
pip install numpy opencv-python matplotlib
```

**Ausführung:**
```bash
git clone https://github.com/3011Alex/CV_Praxis.git
cd CV_Praxis/Präsentation/Anisotropischer\ Filter\ Vergleich
python comparison.py
```

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