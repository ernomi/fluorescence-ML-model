import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import matplotlib
matplotlib.use("QtAgg")  # Automatic Qt binding

# Universal Qt backend for PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QTextEdit,
    QSpinBox, QSlider, QMessageBox, QHBoxLayout, QProgressBar, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Core training function
from core.s_img_data_model import train_models
# Prediction utility
from core.predict_spores import predict_image


# -----------------------------
# Background training thread
# -----------------------------
class TrainingThread(QThread):
    epoch_signal = pyqtSignal(int, float, float)  # epoch, loss, accuracy
    finished_signal = pyqtSignal(dict)           # final results
    batch_signal = pyqtSignal(int)               # global progress in percent

    def __init__(self, data_csv, image_dir, epochs=10, batch_size=32):
        super().__init__()
        self.data_csv = data_csv
        self.image_dir = image_dir
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        # Epoch-level callback
        def emit_epoch(epoch, loss, acc):
            self.epoch_signal.emit(epoch, loss, acc)

        # Batch-level callback
        def emit_batch(global_percent):
            self.batch_signal.emit(int(global_percent))

        results = train_models(
            data_csv=self.data_csv,
            image_dir=self.image_dir,
            epochs=self.epochs,
            batch_size=self.batch_size,
            save_model_dir=str(PROJECT_ROOT / "models"),
            emit_epoch=emit_epoch,
            emit_batch=emit_batch
        )
        self.finished_signal.emit(results)


# -----------------------------
# Main GUI Application
# -----------------------------
class MLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fluorescence ML Model GUI")
        self.setGeometry(200, 200, 950, 700)
        self._threads = []

        # Training history
        self.loss_history = []
        self.acc_history = []

        self.setup_ui()
        self.apply_styles()

    # -----------------------------
    # UI Setup
    # -----------------------------
    def setup_ui(self):
        layout = QVBoxLayout()

        # --- Synthetic Image Generation ---
        layout.addWidget(QLabel("Generate Synthetic Images"))
        self.num_images = QSpinBox()
        self.num_images.setRange(10, 2000)
        self.num_images.setValue(500)
        self.intensity = QSlider(Qt.Orientation.Horizontal)
        self.intensity.setRange(50, 255)
        self.intensity.setValue(210)
        layout.addWidget(QLabel("Number of images per class:"))
        layout.addWidget(self.num_images)
        layout.addWidget(QLabel("Fluorescence intensity:"))
        layout.addWidget(self.intensity)

        self.btn_gen_images = QPushButton("Generate Images")
        self.btn_gen_images.clicked.connect(self.run_gen_images)
        layout.addWidget(self.btn_gen_images)

        # --- Sequence CSV Generation ---
        layout.addSpacing(10)
        layout.addWidget(QLabel("Generate Sequence Data (.csv)"))
        self.num_samples = QSpinBox()
        self.num_samples.setRange(100, 2000)
        self.num_samples.setValue(500)
        layout.addWidget(QLabel("Number of samples:"))
        layout.addWidget(self.num_samples)

        self.btn_gen_seq = QPushButton("Generate CSV")
        self.btn_gen_seq.clicked.connect(self.run_gen_sequences)
        layout.addWidget(self.btn_gen_seq)

        # --- Train Model ---
        layout.addSpacing(10)
        layout.addWidget(QLabel("Train Model"))
        hbox_train_params = QHBoxLayout()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        hbox_train_params.addWidget(QLabel("Epochs:"))
        hbox_train_params.addWidget(self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(32)
        hbox_train_params.addWidget(QLabel("Batch Size:"))
        hbox_train_params.addWidget(self.batch_spin)

        layout.addLayout(hbox_train_params)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Train button
        self.btn_train = QPushButton("Train CNN + Random Forest")
        self.btn_train.clicked.connect(self.run_training)
        layout.addWidget(self.btn_train)

        # --- Predict on new image ---
        layout.addSpacing(10)
        layout.addWidget(QLabel("Predict Spores in Image"))

        self.btn_load_image = QPushButton("Select Image for Prediction")
        self.btn_load_image.clicked.connect(self.load_and_predict_image)
        layout.addWidget(self.btn_load_image)

        # --- Console log ---
        layout.addSpacing(10)
        layout.addWidget(QLabel("Console Log"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output, stretch=1)

        # --- Live training plots ---
        self.plot_canvas = FigureCanvas(Figure(figsize=(7, 3)))
        layout.addWidget(self.plot_canvas)
        self.ax_loss = self.plot_canvas.figure.add_subplot(121)
        self.ax_acc = self.plot_canvas.figure.add_subplot(122)
        self.ax_loss.set_title("Loss")
        self.ax_acc.set_title("Accuracy")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_acc.set_ylabel("Accuracy (%)")
        self.plot_canvas.figure.tight_layout()

        self.setLayout(layout)

    # -----------------------------
    # Styling
    # -----------------------------
    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #202020;
                color: #f0f0f0;
                font-family: Segoe UI, Arial;
                font-size: 11pt;
            }
            QPushButton {
                background-color: #005f73;
                color: white;
                border-radius: 6px;
                padding: 6px 8px;
            }
            QPushButton:pressed { background-color: #054f59; }
            QTextEdit {
                background-color: #1f1f1f;
                color: #e8e8e8;
                border: 1px solid #333;
            }
            QSpinBox, QSlider { margin: 4px 0; }
        """)

    # -----------------------------
    # Button callbacks
    # -----------------------------
    def run_gen_images(self):
        import subprocess
        script = PROJECT_ROOT / "core" / "s_gen_images.py"
        args = [
            sys.executable, str(script),
            "--num_images", str(self.num_images.value()),
            "--intensity", str(self.intensity.value())
        ]
        self._run_process(args, self.btn_gen_images)

    def run_gen_sequences(self):
        import subprocess
        script = PROJECT_ROOT / "core" / "s_gen_sequences.py"
        args = [sys.executable, str(script), "--num_samples", str(self.num_samples.value())]
        self._run_process(args, self.btn_gen_seq)

    def run_training(self):
        self.btn_train.setDisabled(True)
        self.log_output.append("\nTraining started...\n")

        # Reset history
        self.loss_history.clear()
        self.acc_history.clear()
        self.progress_bar.setValue(0)
        self.ax_loss.cla()
        self.ax_acc.cla()
        self.ax_loss.set_title("Loss")
        self.ax_acc.set_title("Accuracy")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_acc.set_ylabel("Accuracy (%)")
        self.plot_canvas.draw()

        thread = TrainingThread(
            data_csv=str(PROJECT_ROOT / "s_data.csv"),
            image_dir=str(PROJECT_ROOT / "s_images"),
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value()
        )
        thread.epoch_signal.connect(self.update_plot)
        thread.batch_signal.connect(self.update_progress_bar)
        thread.finished_signal.connect(self.training_finished)
        self._threads.append(thread)
        thread.start()

    # -----------------------------
    # Predict image using saved CNN
    # -----------------------------
    def load_and_predict_image(self):
        img_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", str(PROJECT_ROOT), "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not img_path:
            return

        try:
            result = predict_image(img_path)
            self.log_output.append(f"Prediction: {result}\n")
            QMessageBox.information(self, "Prediction Result", f"The image contains: {result}")
        except Exception as e:
            self.log_output.append(f"Prediction failed: {e}\n")

    # -----------------------------
    # Update plot per epoch
    # -----------------------------
    def update_plot(self, epoch, loss, acc):
        self.loss_history.append(loss)
        self.acc_history.append(acc)

        self.ax_loss.cla()
        self.ax_acc.cla()
        self.ax_loss.plot(range(1, len(self.loss_history)+1), self.loss_history, 'r-o', label="Loss")
        self.ax_acc.plot(range(1, len(self.acc_history)+1), self.acc_history, 'g-o', label="Accuracy")
        self.ax_loss.set_title("Loss")
        self.ax_acc.set_title("Accuracy")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_acc.set_ylabel("Accuracy (%)")
        self.ax_loss.legend()
        self.ax_acc.legend()
        self.plot_canvas.draw()

        self.log_output.append(f"Epoch {epoch} - Loss: {loss:.4f}, Acc: {acc:.2f}%")
        self.log_output.moveCursor(self.log_output.textCursor().MoveOperation.End)

    # -----------------------------
    # Update progress bar per batch
    # -----------------------------
    def update_progress_bar(self, percent):
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(percent)

    # -----------------------------
    # Training finished
    # -----------------------------
    def training_finished(self, results):
        self.log_output.append("\nTraining finished!\n")
        self.btn_train.setDisabled(False)
        self.progress_bar.setValue(100)
        final_acc = results.get('final_cnn_acc', 'N/A')
        QMessageBox.information(
            self,
            "Training Finished",
            f"Training complete!\nFinal CNN accuracy: {final_acc:.2f}%"
        )

    # -----------------------------
    # Run external scripts
    # -----------------------------
    def _run_process(self, args, button):
        import subprocess
        button.setDisabled(True)
        self.log_output.append(f"\nRunning: {' '.join(args)}\n")
        try:
            subprocess.run(args, check=True)
            self.log_output.append("\nFinished process.\n")
        except subprocess.CalledProcessError as e:
            self.log_output.append(f"\nProcess failed: {e}\n")
        button.setDisabled(False)


# -----------------------------
# Run application
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLApp()
    window.show()
    sys.exit(app.exec())
