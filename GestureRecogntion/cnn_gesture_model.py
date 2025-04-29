import os
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score

# ==== Utility Functions ====
def plot_training_history(history):
    os.makedirs("runs", exist_ok=True)
    epochs = range(1, len(history['accuracy']) + 1)

    plt.figure()
    plt.plot(epochs, history['accuracy'], 'o-', label='Train Acc')
    plt.plot(epochs, history['val_accuracy'], 's--', label='Val Acc')
    for i, acc in enumerate(history['accuracy']):
        plt.text(i + 1, acc, f"{acc:.2f}", ha='center')
    for i, acc in enumerate(history['val_accuracy']):
        plt.text(i + 1, acc, f"{acc:.2f}", ha='center')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("runs/accuracy_plot.png")

    plt.figure()
    plt.plot(epochs, history['loss'], 'o-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 's--', label='Val Loss')
    for i, loss in enumerate(history['loss']):
        plt.text(i + 1, loss, f"{loss:.2f}", ha='center')
    for i, loss in enumerate(history['val_loss']):
        plt.text(i + 1, loss, f"{loss:.2f}", ha='center')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("runs/loss_plot.png")


def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("runs/confusion_matrix.png")


def plot_precision_recall(y_true, y_probs, class_labels):
    for i, label in enumerate(class_labels):
        precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_probs[:, i])
        ap_score = average_precision_score((y_true == i).astype(int), y_probs[:, i])
        plt.figure()
        plt.plot(recall, precision, marker='.', label=f'AP={ap_score:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve: {label}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"runs/precision_recall_{label}.png")


# ==== Main Training Script ====
def main():
    print("🧠 Starting MobileNetV2 Gesture Training...")

    data_dir = 'gesture_data_split'
    img_height, img_width = 224, 224
    batch_size = 32

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    val_data = datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    test_data = datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    num_classes = len(train_data.class_indices)
    class_labels = list(train_data.class_indices.keys())

    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,
        callbacks=[early_stop]
    )

    model.save("gesture_classifier_model.h5")
    print("💾 Model saved as gesture_classifier_model.h5")

    with open("runs/gesture_training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    with open("runs/class_labels.txt", "w") as f:
        for label in class_labels:
            f.write(f"{label}\n")

    plot_training_history(history.history)

    y_probs = model.predict(test_data, verbose=0)
    y_pred_classes = np.argmax(y_probs, axis=1)
    y_true = test_data.classes

    plot_confusion_matrix(y_true, y_pred_classes, class_labels)
    plot_precision_recall(y_true, y_probs, class_labels)

    report_path = "runs/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_true, y_pred_classes, target_names=class_labels))

    print("🎉 All artifacts saved in 'runs/' folder.")


if __name__ == '__main__':
    main()
