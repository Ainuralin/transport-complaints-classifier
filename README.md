# Transport Complaints Classifier

This repository contains a simple NLP pipeline for classifying public transport complaints as positive or negative based on review text written in Russian and Kazakh. It consists of two Python scripts:

- **`1.py`**: Preprocesses and labels the dataset using keyword-based sentiment tagging.
- **`p.py`**: Loads the labeled data, performs exploratory analysis, and visualizes results (confusion matrix, class distribution, word frequencies).

## 📂 Project Structure

```
transport-complaints-classifier/
├── AI_dataset.xlsx            # Original dataset (reviews in Russian & Kazakh)
├── AI_dataset_labeled.xlsx    # Output after keyword-based labeling
├── 1.py                       # Script for labeling reviews
├── p.py                       # Script for analysis & visualization
├── README.md                  # Project overview and instructions
└── requirements.txt           # Python dependencies
```

## 🛠 Prerequisites

- Python 3.7+
- pip

All required Python packages are listed in `requirements.txt`.

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/transport-complaints-classifier.git
   cd transport-complaints-classifier
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1. Labeling Reviews (`1.py`)

This script reads `AI_dataset.xlsx`, converts texts to lowercase, and assigns sentiment labels:
- `1` for positive reviews (matches positive keywords in Russian/Kazakh).
- `0` for negative reviews (matches negative keywords).

To run:
```bash
python 1.py
```
This generates `AI_dataset_labeled.xlsx` with an added `label` column.

### 2. Analysis and Visualization (`p.py`)

This script loads `AI_dataset_labeled.xlsx`, filters out unlabeled entries, and:
- Splits data into training and test sets.
- (Optional) Demonstrates baseline keyword labeling with a confusion matrix.
- Visualizes class distribution (positive vs. negative).
- Preprocesses text (regex cleaning, stopword removal, lemmatization).
- Plots top 10 word frequencies for each sentiment.

To run:
```bash
python p.py
```

The script displays:
- Confusion matrix of keyword-based baseline.
- Bar chart of class distribution.
- Bar plots of the most frequent words per class.

## 📈 Results

- **Keyword-based labeling**: ~48 reviews labeled (44 negative, 4 positive).
- **Baseline accuracy**: approximately 90% on test split using keyword labels.
- Visualizations help understand data imbalance and common terms.

## 🖥 Video Demonstration

A short video demo is available at: `<YOUR_VIDEO_LINK>`

### What to include:
1. Loading data and showing sample reviews.
2. Running `1.py` and examining labeled output.
3. Running `p.py` to display confusion matrix and charts.
4. Brief summary of findings.

## 🔧 Future Improvements

- Expand keyword lists to improve labeling recall.
- Train a proper machine learning classifier (e.g., Naive Bayes, Logistic Regression, Transformer-based) on the labeled data.
- Balance the dataset by adding more positive examples.
- Integrate into a web app or dashboard for real-time monitoring.

## 🤝 Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

