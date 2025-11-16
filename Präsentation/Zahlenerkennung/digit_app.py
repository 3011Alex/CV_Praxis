import tkinter as tk
from tkinter import *
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras
import os

MODEL_PATH = "mnist_model.h5"


# Laden oder trainieren des MNIST-Modells

def get_model():
    if os.path.exists(MODEL_PATH):
        print("Lade vorhandenes Modell...")
        return keras.models.load_model(MODEL_PATH)

    print("Kein Modell gefunden – trainiere MNIST Modell...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    # CNN-Modell für 0-9 Erkennung
    model = keras.Sequential([
        # Block 1
        keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3,3), activation="relu", padding='same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.25),
        
        # Block 2
        keras.layers.Conv2D(64, (3,3), activation="relu", padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3,3), activation="relu", padding='same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.25),
        
        # Block 3
        keras.layers.Conv2D(128, (3,3), activation="relu", padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3,3), activation="relu", padding='same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.25),
        
        # Dense Layers
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation="softmax")
    ])

    # Optimaler Optimizer
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    # Callbacks für besseres Training
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5, 
        patience=3, 
        min_lr=0.00001, 
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        patience=8, 
        restore_best_weights=True, 
        verbose=1
    )

    print("Training läuft...")
    
    # Intensives Training für maximale Präzision
    history = model.fit(
        x_train, y_train,
        epochs=10,  # Kürzer aber effektiv
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=[reduce_lr, early_stop],
        verbose=1
    )
    
    # Detaillierte Evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\Finale Test-Genauigkeit: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Per-Klasse Genauigkeit prüfen
    predictions = model.predict(x_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    from sklearn.metrics import classification_report
    print("\nGenauigkeit pro Ziffer (0-9):")
    print(classification_report(y_test, predicted_classes, 
                              target_names=[f"Ziffer {i}" for i in range(10)],
                              digits=4))
    
    model.save(MODEL_PATH)
    print(f"\nModell gespeichert als: {MODEL_PATH}")

    return model


model = get_model()


# Interface
class App:
    def __init__(self, master):
        self.master = master
        master.title("Zahlenerkennung")
        master.configure(bg='lightgray')

        # Größeres Canvas für bessere Zeichenqualität
        self.canvas = Canvas(master, width=400, height=400, bg="white", bd=2, relief="solid")
        self.canvas.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.paint)

        self.button_frame = Frame(master)
        self.button_frame.pack(pady=5)

        Button(self.button_frame, text="🔍 Erkennen", command=self.predict_digit, 
               font=("Arial", 12), bg="lightblue", width=12).pack(side=LEFT, padx=5)
        Button(self.button_frame, text="🗑️ Löschen", command=self.clear, 
               font=("Arial", 12), bg="lightcoral", width=12).pack(side=LEFT, padx=5)

        self.label = Label(master, text="Ergebnis: -\nTop 3: -", font=("Arial", 14), justify="center", 
                          bg='lightgray', fg='darkblue')
        self.label.pack(pady=10)

    def paint(self, event):
        # Dickere Pinselstriche für bessere Erkennung
        x1, y1 = event.x - 15, event.y - 15
        x2, y2 = event.x + 15, event.y + 15
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")

    def clear(self):
        self.canvas.delete("all")
        self.label.config(text="Ergebnis: -\nTop 3: -")

    def predict_digit(self):     
        # Canvas-Inhalt in ein PIL-Image zeichnen
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Neues weißes Bild erstellen
        img = Image.new('RGB', (canvas_width, canvas_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Alle Canvas-Objekte durchgehen und nachzeichnen
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) == 4:  # Oval/Kreis
                x1, y1, x2, y2 = coords
                draw.ellipse([x1, y1, x2, y2], fill='black')
        
        # Zu Graustufen und 28x28 konvertieren
        img = img.convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img)

        # Debug: Originalbild speichern
        debug_orig = Image.fromarray(img_array)
        debug_orig.save("debug_original.png")
        print(f"Debug: Original image size: {img_array.shape}, min: {img_array.min()}, max: {img_array.max()}")

        # Invertieren: Weiß (255) wird schwarz (0), schwarz (0) wird weiß (255)
        img_array = 255 - img_array
        
        # Debug: Invertiertes Bild speichern
        debug_inv = Image.fromarray(img_array.astype(np.uint8))
        debug_inv.save("debug_inverted.png")
        print(f"Debug: Inverted image size: {img_array.shape}, min: {img_array.min()}, max: {img_array.max()}")
        
        # Normalisierung für Neural Network
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Prediction mit Konfidenz
        prediction = model.predict(img_array, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Debug-Info ausgeben
        print(f"Prediction Array: {prediction[0]}")
        print(f"Erkannte Ziffer: {digit}, Konfidenz: {confidence:.1f}%")
        
        # Alle Top-3 Vorhersagen anzeigen
        top_3_indices = np.argsort(prediction[0])[::-1][:3]
        top_3_text = " | ".join([f"{i}: {prediction[0][i]*100:.1f}%" for i in top_3_indices])

        self.label.config(text=f"Ergebnis: {digit} ({confidence:.1f}%)\nTop 3: {top_3_text}")

root = tk.Tk()
app = App(root)
root.mainloop()
