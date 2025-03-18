import json
import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


# 2. Загрузка и подготовка данных
class PharmacyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            for item in raw_data:
                text = f"Вопрос: {item['Вопрос']}\nОтвет: {item['Ответ']}"
                self.data.append(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {key: val.squeeze() for key, val in encoding.items()}


# 3. Инициализация модели и токенизатора
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Инициализация устройства после создания модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. Загрузка датасета
dataset = PharmacyDataset(
    file_path="D:/DEV/HagAss/pharmacy_data.json",
    tokenizer=tokenizer
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Для CausalLM задачи
)
# 5. Разделение данных
train_size = int(0.9 * len(dataset))
train_dataset = Subset(dataset, range(train_size))
val_dataset = Subset(dataset, range(train_size, len(dataset)))

# 6. Настройка обучения
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=7,
    logging_dir="./logs",
    save_steps=500,
    eval_strategy="steps",  # Было evaluation_strategy
    eval_steps=500,
    logging_steps=100,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# 7. Запуск обучения
trainer.train()


# 8. Функции для генерации ответов
def generate_response(user_input):
    inputs = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt",
        padding='max_length',  # Добавлено
        max_length=512,  # Добавлено
        truncation=True  # Добавлено
    ).to(device)  # Добавлено перемещение на устройство

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],  # Ключевое изменение
        max_length=500,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=82,
        top_p=0.85,
        temperature=0.7,
        repetition_penalty=1.2  # Штраф за повторения
    )

    #return tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return tokenizer.decode(outputs[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)

def process_input(text):
    emergency_keywords = ["скорая", "умираю", "кровь", "потерял сознание"]
    if any(kw in text.lower() for kw in emergency_keywords):
        return "⚠️ Срочно вызовите скорую помощь (103)! После оказания помощи мы поможем с лекарствами."

    if any(kw in text.lower() for kw in ["доставка", "город", "заказать"]):
        return phase_1_sales(text)

    return generate_response(text)


def phase_1_sales(text):
    return "🏪 Для оформления заказа укажите:\n1. Город\n2. Название препарата\n3. Способ получения (доставка/самовывоз)"


# 9. Тестирование и сохранение модели
if __name__ == "__main__":
    # Сохранение модели после обучения
    model.save_pretrained("./pharmacy_gpt")
    tokenizer.save_pretrained("./pharmacy_gpt")

    # Тестирование
    while True:
        user_input = input("Вы: ")
        if user_input.lower() in ["exit", "выход"]:
            break
        print("Аптекарь:", process_input(user_input))