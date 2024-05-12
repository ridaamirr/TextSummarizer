import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTextEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

from transformers import BartForConditionalGeneration, BartTokenizer

class TextSummarizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Summarizer")
        self.setGeometry(100, 100, 600, 400)

        self.setup_ui()
        
        # Load the tokenizer and model
        model_path = "D:/Semester 6/AI/TextSummarizer/Custom_BART_Model-20240511T202739Z-001/Custom_BART_Model"
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path)

        self.setStyleSheet(
            "QMainWindow { background-color: #1E1E1E; color: #FFFFFF; }"
            "QLabel { color: #FFFFFF; font-size: 16px; }"  # Removed font-weight: bold
            "QTextEdit { background-color: #333333; color: #FFFFFF; border: 2px solid #28a871; font-size: 16px; }"
            "QPushButton { background-color: #28a871; color: #FFFFFF; border: 2px solid #28a871; border-radius: 4px; font-size: 18px; }"
            "QPushButton:hover { background-color:#00cc66 ; border-color: #FFFFFF; }"
        )

    def setup_ui(self):
        central_widget = QWidget()
        layout = QVBoxLayout()

        self.source_label = QLabel("Enter text to summarize:")
        layout.addWidget(self.source_label)

        self.source_text_area = QTextEdit()
        self.source_text_area.textChanged.connect(self.update_word_count)
        layout.addWidget(self.source_text_area)

        self.word_count_label = QLabel("Word Count: 0")
        self.word_count_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        layout.addWidget(self.word_count_label, 0, Qt.AlignRight | Qt.AlignBottom)

        self.summarize_button = QPushButton("Summarize")
        self.summarize_button.clicked.connect(self.summarize_text)
        layout.addWidget(self.summarize_button)

        self.target_label = QLabel("Summary:")
        layout.addWidget(self.target_label)

        self.target_text_area = QTextEdit()
        self.target_text_area.setReadOnly(True)  # Make summary box read-only
        layout.addWidget(self.target_text_area)

        self.summary_word_count_label = QLabel("Summary Word Count: 0")
        self.summary_word_count_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        layout.addWidget(self.summary_word_count_label, 0, Qt.AlignRight | Qt.AlignBottom)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def update_word_count(self):
        text = self.source_text_area.toPlainText()
        word_count = len(text.split())
        self.word_count_label.setText(f"Word Count: {word_count}")

    def summarize_text(self):
        input_text = self.source_text_area.toPlainText()

        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)

        # Generate the summary
        summary_ids = self.model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Update the target text area with the summary
        self.target_text_area.setPlainText(summary)

        # Update the word count for the summary
        summary_word_count = len(summary.split())
        self.summary_word_count_label.setText(f"Summary Word Count: {summary_word_count}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextSummarizerApp()
    window.show()
    sys.exit(app.exec_())
