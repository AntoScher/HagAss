import json
import re


def convert_txt_to_json(txt_path, output_json_path):
    """Конвертирует текстовый файл с вопросами-ответами в структурированный JSON"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Улучшенное регулярное выражение для разных форматов
    qa_pattern = re.compile(
        r'(?:\n|^)\s*([^\n:?]+)\s*[:?]\s*([^\n]+?)\s*(?=\n\s*[^\n:?]+\s*[:?]|$)',
        re.MULTILINE
    )

    matches = qa_pattern.findall(content)

    data = []
    errors = []

    for idx, (question, answer) in enumerate(matches):
        # Нормализация текста
        question = question.strip()
        answer = answer.strip()

        # Проверка на пустые значения
        if not question:
            errors.append(f"ОШИБКА: Пустой вопрос в записи {idx}")
            continue

        if not answer:
            errors.append(f"ОШИБКА: Пустой ответ в записи {idx} (вопрос: '{question}')")
            continue

        data.append({
            "Вопрос": question,
            "Ответ": answer
        })

    # Сохранение результатов
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Вывод отладочной информации
    print(f"\nОбработано записей: {len(matches)}")
    print(f"Успешно конвертировано: {len(data)}")
    print(f"Найдено ошибок: {len(errors)}")

    if errors:
        print("\nСписок ошибок:")
        print("\n".join(errors[:5]))  # Показываем первые 5 ошибок для примера

    print(f"\nРезультат сохранен в: {output_json_path}")


if __name__ == "__main__":
    convert_txt_to_json(
        txt_path='pharmacy_data.txt',
        output_json_path='pharmacy_data.json'
    )