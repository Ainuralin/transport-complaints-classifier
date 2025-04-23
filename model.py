import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

df_labeled = pd.read_excel("AI_dataset_labeled.xlsx")
df_labeled.dropna(subset=['label', 'Описание'], inplace=True)
df_labeled['label'] = df_labeled['label'].astype(int)
df_labeled['text_lower'] = df_labeled['Описание'].astype(str).str.lower()

X_train, X_test, y_train, y_test = train_test_split(df_labeled['text_lower'], df_labeled['label'], test_size=0.3, random_state=42)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words_ru = set(nltk.corpus.stopwords.words('russian'))
stop_words_kz = set()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^а-яёқғңұүһіА-ЯЁҚҒҢҰҮҺІ\s]', '', text)
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words_ru and word not in stop_words_kz]
        return " ".join(words)
    return ""

df_labeled['text_cleaned'] = df_labeled['Описание'].astype(str).apply(preprocess_text)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nМатрица ошибок:\n", confusion_matrix(y_test, y_pred))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Отрицательный', 'Положительный'], yticklabels=['Отрицательный', 'Положительный'])
plt.title('Матрица ошибок (с использованием Naive Bayes)')
plt.ylabel('Фактические метки')
plt.xlabel('Предсказанные метки')
plt.show()

print("\nОтчет о классификации:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df_labeled)
plt.title('Распределение тональности отзывов')
plt.xlabel('Метка (0: Отрицательный, 1: Положительный)')
plt.ylabel('Количество отзывов')
plt.show()

positive_reviews = " ".join(df_labeled[df_labeled['label'] == 1]['text_cleaned'].dropna())
negative_reviews = " ".join(df_labeled[df_labeled['label'] == 0]['text_cleaned'].dropna())

positive_word_counts = Counter(positive_reviews.split())
negative_word_counts = Counter(negative_reviews.split())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x=[word for word, count in positive_word_counts.most_common(10)], y=[count for word, count in positive_word_counts.most_common(10)])
plt.title('Топ 10 слов в положительных отзывах')
plt.xlabel('Слово')
plt.ylabel('Частота')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.subplot(1, 2, 2)
sns.barplot(x=[word for word, count in negative_word_counts.most_common(10)], y=[count for word, count in negative_word_counts.most_common(10)])
plt.title('Топ 10 слов в отрицательных отзывах')
plt.xlabel('Слово')
plt.ylabel('Частота')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()

print("Графики матрицы ошибок, распределения классов и частоты слов отображены.")
