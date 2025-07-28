# Прогнозирование целевых действий пользователей на платформе «СберАвтоподписка»

## 📌 Описание проекта

Учебный проект, выполненный в рамках практики магистратуры ТГУ по направлению **«Науки о данных и машинное обучение»**.  
Цель — разработка модели машинного обучения для предсказания вероятности совершения пользователем целевого действия на сайте сервиса **СберАвтоподписка**.

Тип задачи: бинарная классификация  
Метрика: `ROC-AUC`

---

## 👥 Команда проекта

- **Тимлид**: Гришин Сергей  
- **Команда**:
  - Вишняков Дмитрий  
  - Данилова Елена  
  - Коваленко Екатерина  
  - Тагильцев Кирилл  
  - Шерин Иван  

---



## 📁 Структура проекта

- [1. Описательный анализ данных, портрет пользователя, посещаемость](https://github.com/servgri/hackaton_2/blob/master/Notes/1.%20%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9%20%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85%2C%20%D0%BF%D0%BE%D1%80%D1%82%D1%80%D0%B5%D1%82%20%D0%BF%D0%BE%D0%BB%D1%8C%D0%B7%D0%BE%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D1%8F%2C%20%D0%BF%D0%BE%D1%81%D0%B5%D1%89%D0%B0%D0%B5%D0%BC%D0%BE%D1%81%D1%82%D1%8C.ipynb)
- [2. Формирование датафрейма для обучения](https://github.com/servgri/hackaton_2/blob/master/Notes/2.%20%D0%A4%D0%BE%D1%80%D0%BC%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_%D0%B4%D0%B0%D1%82%D0%B0%D1%84%D1%80%D0%B5%D0%B9%D0%BC%D0%B0_%D0%B4%D0%BB%D1%8F_EDA.ipynb)
- [3. Разведочный анализ данных](https://github.com/servgri/hackaton_2/blob/master/Notes/2.%20EDA.ipynb)
- [4. Pipeline обучения](https://github.com/servgri/hackaton_2/blob/master/Notes/2.%20%D0%A4%D0%BE%D1%80%D0%BC%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_%D0%B4%D0%B0%D1%82%D0%B0%D1%84%D1%80%D0%B5%D0%B9%D0%BC%D0%B0_%D0%B4%D0%BB%D1%8F_EDA.ipynb)
- [5. Отбор признаков](https://github.com/servgri/hackaton_2/blob/master/Notes/4.%20feature_selection.ipynb)
- [6. Обучение нейронной сети](https://github.com/servgri/hackaton_2/blob/master/Notes/4.%20feature_selection.ipynb)
- [7. Тестирование API на Flask](https://github.com/servgri/hackaton_2/blob/master/Notes/6.%20Test_api.ipynb)
- [Дополнительно: модели](https://drive.google.com/drive/folders/1CiQLk7Q3e_nQpS8ZKRH1JOewRrTW-rio?usp=drive_link)
- [Дополнительно: датафреймы](https://drive.google.com/drive/folders/1Rwh1nT6mf-6_3dBtApRE7zBziY_ocnsC?usp=drive_link)
---

## 🎯 Цель проекта

Разработать модель, предсказывающую вероятность того, что пользователь совершит одно из целевых действий:
- «Оставить заявку»
- «Заказать звонок»

---

## 🗃 Источник данных

Использованы логи пользовательской активности на сайте:
- `utm_*` — рекламные метки
- `device_*` — характеристики устройств
- `geo_*` — геоданные
- `visit_*` — информация о визите
- `event_*`, `hit_*` — события на сайте

📁 [Скачать данные](https://cloud.mail.ru/public/PXoc/hDmWMRLe6)  
📄 [Полное задание](https://lms-cdn.skillfactory.ru/assets/courseware/v1/d71c2fe9706361f6010e7d05243fb4a2/asset-v1:skillfactory+TGUDS-2sem+2025+type@asset+block/%D0%A3%D1%87%D0%B5%D0%B1%D0%BD%D0%B0%D1%8F_%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B0_%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7_%D1%81%D0%B0%D0%B9%D1%82%D0%B0.docx)

---

## 🔧 Этапы работы

### 1. Предобработка данных
- Обработка пропусков и редких категорий
- Категоризация источников трафика
- Инженерия признаков по дате и времени визита
- Кодирование категориальных признаков (`TopEncoderTransformer`, `MapColumnTransformer`)

### 2. Сборка пайплайна
- Кастомные `sklearn`-трансформеры
- `ColumnTransformer` с `StandardScaler` и `OneHotEncoder`
- Объединение всего в единый `Pipeline`

### 3. Обучение моделей
- Подбор гиперпараметров с помощью `GridSearchCV`
- Использованные модели:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Extra Trees
  - Gradient Boosting
  - LightGBM
  - XGBoost
  - CatBoost
  - HistGradientBoosting
  - BaggingClassifier

### 4. Оценка результатов
- Метрика: ROC-AUC
- Визуализация: ROC-кривые и графики важности признаков
- Сравнение всех моделей

### 5. Дополнительно
- Обучение простой нейронной сети
- Сохранение моделей с помощью `pickle`

---

## 📈 Результаты

- Лучшие модели достигли ROC-AUC **> 0.75**
- Удовлетворены требования:
  - Качество: ROC-AUC > 0.65
  - Скорость: время предсказания ≤ 3 секунд
- Важнейшие признаки: `utm_source`, `device_type`, `geo_city`, день недели и др.

---

## 🛠 Используемые технологии

- Python 3.10
- Pandas, NumPy
- Scikit-learn
- LightGBM, XGBoost, CatBoost
- Matplotlib, Seaborn
- Google Colab / Jupyter Notebook
- Pickle (сохранение моделей)

---

## 🚗 О сервисе «СберАвтоподписка»

**«СберАвтоподписка»** — сервис долгосрочной аренды автомобилей для физических лиц.

Включает:
- Ежемесячный фиксированный платеж
- Страхование (КАСКО, ОСАГО, ДСАГО)
- ТО, ремонт, смена и хранение шин
- 24/7 поддержка
- Доп. опция: консьерж-сервис

---

## 🧾 Лицензия

Проект выполнен в учебных целях. Использование и распространение ограничено образовательной задачей.
