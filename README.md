## 📄 AI‑поисковик по PDF (Streamlit)

Задаёшь вопрос — получаешь ответ из загруженных PDF. Под капотом: LangChain + векторное хранилище (Chroma/FAISS) + OpenAI или локальный Ollama.

### Возможности
- Загрузка нескольких PDF
- Индексация текста (чанки + embeddings)
- Поиск по контексту с указанием источников/страниц
- Провайдеры: OpenAI или Ollama
- Векторные БД: Chroma (с опциональным сохранением на диск) или FAISS (в памяти)

### Установка
```powershell
cd "Your Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Переменные окружения
Создайте файл `PDF.env` (или `.env`) по примеру ниже:
```env
OPENAI_API_KEY=sk-your-openai-api-key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b-instruct
```

### Запуск
```powershell
streamlit run PDF.py
```

Загрузите PDF‑файлы, нажмите «Построить индекс», затем задавайте вопросы.

### Примечания
- Для локального режима установите Ollama (`https://ollama.com`) и скачайте модель: `ollama pull llama3:8b-instruct`
- Для сохранения Chroma укажите каталог и включите опцию «Сохранять индекс на диск»

