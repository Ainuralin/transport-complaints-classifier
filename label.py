import pandas as pd
import numpy as np

df = pd.read_excel("AI_dataset.xlsx")

df['text_lower'] = df['Описание'].astype(str).str.lower()

positive_keywords = [
    "жақсы", "ұнады", "керемет", "күшті", "тамаша", "қанағат",
    "хорошо", "понравилось",
    "спасибо", "удобно", "нравится", "быстро", "отлично","тез келді",
    "қанағаттанарлық", "хорошо", "пришёл вовремя",
    "приехал быстро", "работает отлично", "удовлетворительно"
]

negative_keywords = [
    "жаман", "кешікті", "жоқ", "нашар", "күту", "дұрыс емес", "кешігу", "есікті ашпады",
    "шағым", "күттім", "кешігіп", "плохо", "опоздание", "опоздал", "кетіп", "шара",
    "өтіп кетті", "жауап бермеді", "сасық", "қатар келді",
    "проблема", "долго", "жалоба", "ожидание",
    "келмеді", "тоқтамады", "уақытында келмеді", "жол ұзақ", "жол нашар", "срочно", "ожидает",
    "болмады", "не", "долго ждал", "вышел из строя", "опоздал", "время ожидания", "сняли оплату", "меры", "сильный запах",
    "ожидают", "мало", "проехал", "медленно", "закрыл", "файлов"
]

def classify_sentiment(text):
    if any(word in text for word in positive_keywords):
        return 1
    elif any(word in text for word in negative_keywords):
        return 0
    else:
        return np.nan

df['label'] = df['text_lower'].apply(classify_sentiment)

df.to_excel("AI_dataset_labeled.xlsx", index=False)
print("Файл сохранён: AI_dataset_labeled.xlsx")
