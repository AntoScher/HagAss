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

# 1. Загрузка системного промпта
with open('system_prompt.txt', 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read().strip()

# 2. Конфигурационные параметры
MAX_INPUT_LENGTH = 384  # Уменьшено для резерва под генерацию
MAX_NEW_TOKENS = 128  # Максимальная длина ответа


# 3. Класс датасета с учетом лимитов
class PharmacyDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

            for item in raw_data:
                text = f"{SYSTEM_PROMPT}\n\nВопрос: {item['Вопрос']}\nОтвет: {item['Ответ']}"
                self.data.append(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.data[idx],
            max_length=MAX_INPUT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {k: v.squeeze() for k, v in encoding.items()}


# 4. Инициализация модели
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. Подготовка данных
dataset = PharmacyDataset(
    file_path="D:/DEV/HagAss/pharmacy_data.json",
    tokenizer=tokenizer
)
train_size = int(0.9 * len(dataset))
train_dataset = Subset(dataset, range(train_size))
val_dataset = Subset(dataset, range(train_size, len(dataset)))

# 6. Конфигурация обучения
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=7,
    learning_rate=3e-5,
    logging_dir="./logs",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
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

# 7. Обучение
trainer.train()


# 8. Исправленная генерация ответов
def generate_response(user_input):
    # Формируем полный промпт
    full_prompt = f"{SYSTEM_PROMPT}\n\nВопрос: {user_input}\nОтвет:"

    # Кодируем с учетом лимитов
    inputs = tokenizer.encode_plus(
        full_prompt,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        padding='max_length',
        truncation=True
    ).to(device)

    # Генерируем с контролем длины
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=MAX_NEW_TOKENS,  # Ключевое изменение!
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.85,
        top_k=100,
        top_p=0.95,
        repetition_penalty=1.15
    )

    # Вырезаем только ответ
    return tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:],
        skip_special_tokens=True
    )


# 9. Обработка ввода
def process_input(text):
    emergency_keywords = ["скорая", "умираю", "кровь", "потерял сознание"]
    if any(kw in text.lower() for kw in emergency_keywords):
        return "⚠️ Срочно вызовите скорую (103)! После помощи поможем с лекарствами."

    if any(kw in text.lower() for kw in ["доставка", "город", "заказать"]):
        return "🏪 Для заказа укажите:\n1. Город\n2. Препарат\n3. Способ получения"

    response = generate_response(text)
    return response if len(response) > 2 else "Уточните запрос, пожалуйста."


# 10. Запуск системы
if __name__ == "__main__":
    model.save_pretrained("./pharmacy_gpt")
    tokenizer.save_pretrained("./pharmacy_gpt")

    print("Система готова. Для выхода введите 'выход'")
    while True:
        user_input = input("Вы: ")
        if user_input.lower() in ["выход", "exit"]:
            break
        print("Аптекарь:", process_input(user_input))