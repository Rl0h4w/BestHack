# README: Обработка видео для непрерывного предсказания целевых значений

## Обзор

Этот проект реализует модель глубокого обучения для обработки видеоданных и предсказания непрерывных целевых значений, вероятно, связанных с речью или выражениями лица. Модель использует трехмерную сверточную нейронную сеть (CNN) в сочетании со слоями Long Short-Term Memory (LSTM) для захвата как пространственных, так и временных признаков из видеокадров. Обнаружение лиц используется для фокусировки на области лица в каждом кадре, повышая способность модели изучать релевантные признаки.

Код разработан для того, чтобы:

1.  **Обрабатывать видеоданные**: Загружать видеофайлы и соответствующие файлы с целевыми значениями (ground truth).
2.  **Обнаруживать лица**: Использовать YOLOv8-face для обнаружения лиц на каждом видеокадре.
3.  **Выделять области лиц**: Обрезать кадры, чтобы сфокусироваться на обнаруженных областях лиц, улучшая фокусировку и эффективность модели.
4.  **Предварительно обрабатывать кадры**: Изменять размер и нормализовать видеокадры.
5.  **Генерировать пакеты**: Создавать оконные последовательности кадров и целевых значений, подходящие для обучения модели временных рядов.
6.  **Строить 3D CNN-LSTM модель**: Создавать модель глубокого обучения, используя 3D CNN для извлечения пространственных признаков и LSTM для моделирования временных последовательностей.
7.  **Обучать модель**: Обучать модель, используя подготовленный набор данных, с колбэками для мониторинга производительности, сохранения модели и ранней остановки.
8.  **Оценивать производительность**: Оценивать обученную модель на тестовом наборе данных и сообщать о метриках производительности.

Этот README предоставляет подробное руководство по пониманию, настройке и использованию этого проекта.

## Набор данных

### Структура набора данных

Проект разработан для работы с набором данных, организованным в следующей структуре каталогов:

```
data_dir/
    subject_id_1/
        session_id_1/
            vid.avi                # Видеофайл
            ground_truth.txt       # Файл с целевыми значениями (ground truth)
        session_id_2/
            ...
    subject_id_2/
        ...
    ...
```

  * **`data_dir`**: Корневой каталог, содержащий набор данных. Вам нужно будет указать путь к вашим каталогам с обучающими и тестовыми данными при запуске кода. В предоставленном скрипте примеры путей: `/kaggle/input/sxuprl/train/train` и `/kaggle/input/sxuprl/test/test`.
  * **`subject_id_X`**: Каталоги, представляющие отдельных субъектов. Соглашение об именовании этих каталогов может варьироваться в зависимости от вашего набора данных.
  * **`session_id_Y`**: Внутри каждого каталога субъекта находятся каталоги сессий. Опять же, соглашение об именовании зависит от набора данных. Каждый каталог сессии содержит видео и файлы ground truth для определенной сессии записи.
  * **`vid.avi`**: Видеофайл в формате AVI, содержащий видео русской речи.
  * **`ground_truth.txt`**: Текстовый файл, содержащий разделенные пробелами числа с плавающей точкой, представляющие целевые значения для каждого кадра соответствующего видео. Количество значений должно соответствовать количеству кадров в видео.

### Получение набора данных

Предоставленный код настроен на использование набора данных, расположенного в `/kaggle/input/sxuprl/train/train` и `/kaggle/input/sxuprl/test/test`, что предполагает, что он может быть связан с набором данных "SXuPRL" или аналогичным набором данных, доступным на Kaggle.

Если вы намерены использовать другой набор данных, убедитесь, что он соответствует структуре каталогов, описанной выше. Вам нужно будет:

1.  **Получить свой набор данных**: Получите набор данных видео на русском языке с соответствующими файлами ground truth.
2.  **Организовать набор данных**: Структурируйте свой набор данных в формат каталогов, указанный выше.
3.  **Настроить пути к данным**: Измените пути `data_dir` в функции `create_dataset`, чтобы указать на каталоги обучения и тестирования вашего набора данных.

### Формат Ground Truth

Файл `ground_truth.txt` имеет решающее значение. Он должен содержать одну строку чисел с плавающей точкой, разделенных пробелами. Каждое значение соответствует кадру в видеофайле `vid.avi`. Убедитесь, что количество значений в этом файле точно соответствует количеству кадров в видео, чтобы избежать ошибок при загрузке данных. Характер этих целевых значений (что они представляют - например, интенсивность выражения, направление взгляда, речевой признак) зависит от конкретной задачи и набора данных, которые вы используете.

## Структура кода

Код организован в следующие ключевые компоненты:

  * **Файлы Python**: Весь проект содержится в одном файле Python.

  * **Классы**:

      * **`VideoProcessor`**: Этот класс является центральным для загрузки и предварительной обработки данных. Он обрабатывает:
          * Инициализация: Загружает модель обнаружения лиц YOLOv8-face и настраивает буферы кадров и целевых значений.
          * `process_video(video_path, target_path)`: Генерирует пакеты обработанных видеокадров и их соответствующих целевых значений из заданного видео и пути к файлу целевых значений. Он выполняет обнаружение лиц, обрезку кадров, изменение размера, нормализацию и оконную обработку.
          * `_process_batch(yolo_batch, original_frames, targets, start_idx)`: Обрабатывает пакет кадров, используя модель YOLOv8-face для обнаружения лиц, обрезки кадров и добавления обработанных кадров и целевых значений в буферы.

  * **Функции**:

      * `get_video_paths(data_dir)`: Рекурсивно сканирует каталог данных, чтобы найти пары видео (`vid.avi`) и ground truth (`ground_truth.txt`). Возвращает список кортежей, где каждый кортеж содержит пути к видеофайлу и соответствующему файлу целевых значений.
      * `create_dataset(data_dir, num_parallel_calls=PARALLEL_VIDEOS)`: Создает объект TensorFlow `Dataset` из заданного каталога данных. Он использует `tf.data.Dataset.interleave` для параллельной обработки видео с использованием класса `VideoProcessor`, что значительно ускоряет загрузку данных.
      * `build_3dcnn_lstm_model()`: Определяет и компилирует архитектуру 3D CNN-LSTM модели, используя TensorFlow Keras API. Он задает слои, функции активации, регуляризацию (Batch Normalization, Dropout), оптимизатор, функцию потерь и метрики оценки.

  * **Глобальные переменные**:

      * `WINDOW_SIZE = 3`: Определяет количество кадров в каждом входном окне для LSTM-модели.
      * `HEIGHT, WIDTH = 120, 160`: Указывает высоту и ширину, до которых изменяется размер видеокадров после обрезки лица (или изменения размера всего кадра, если лицо не обнаружено).
      * `YOLO_BATCH_SIZE = 256`: Размер пакета для обработки кадров через модель YOLOv8-face. Большие пакеты могут повысить эффективность обработки YOLO, но увеличить использование памяти.
      * `DATA_BATCH_SIZE = 128`: Размер пакета для обучения 3D CNN-LSTM модели.
      * `PARALLEL_VIDEOS = 4`: Количество видео, обрабатываемых параллельно при создании набора данных, используя `tf.data.Dataset.interleave`.

## Зависимости

Проект зависит от следующих библиотек Python:

  * **os**: Для взаимодействия с операционной системой, в основном для манипуляций с путями к файлам.
  * **cv2 (OpenCV)**: Для задач обработки видео и изображений, включая чтение видеофайлов, изменение размера кадров и предварительную обработку для обнаружения лиц.
  * **numpy**: Для численных операций, особенно для обработки видеокадров и целевых значений в виде массивов.
  * **collections.deque**: Для эффективной буферизации кадров и операций с очередями в классе `VideoProcessor`.
  * **tensorflow**: Основная библиотека глубокого обучения, используемая для построения, обучения и оценки 3D CNN-LSTM модели.
  * **ultralytics (YOLO)**: Специально для использования модели обнаружения лиц YOLOv8-face. Убедитесь, что `ultralytics` установлен и что файл весов модели YOLOv8-face (`yolov8n-face.pt`) доступен по указанному пути.
  * **tqdm.keras.TqdmCallback**: Для отображения индикатора прогресса во время эпох обучения модели.

**Установка**:

Рекомендуется создать виртуальное окружение для управления зависимостями. Вы можете установить все необходимые библиотеки, используя `pip`:

```bash
pip install opencv-python numpy tensorflow ultralytics tqdm
```

Для TensorFlow вы можете захотеть установить версию с поддержкой GPU (`tensorflow-gpu`), если у вас есть совместимая видеокарта NVIDIA для ускорения обучения. Обратитесь к документации TensorFlow для получения инструкций по установке, специфичных для вашей системы и оборудования.

## Настройка и использование

### 1\. Настройка окружения

1.  **Установите Python**: Убедитесь, что у вас установлен Python 3.7 или более поздней версии.
2.  **Создайте виртуальное окружение (Рекомендуется)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # В Linux/macOS
    venv\Scripts\activate  # В Windows
    ```
3.  **Установите зависимости**: Используйте команду `pip install`, упомянутую в разделе "Зависимости" выше.

### 2\. Подготовьте набор данных

1.  **Получите набор данных**: Получите набор данных видео на русском языке и файлы ground truth, и организуйте их в соответствии со структурой каталогов, описанной в разделе "Набор данных".
2.  **Разместите набор данных**: Если вы используете Kaggle или аналогичную среду, поместите свой набор данных в место, доступное для вашего скрипта. При локальном запуске убедитесь, что пути к данным в функции `create_dataset` правильно указывают на каталоги вашего набора данных. Для предоставленного скрипта предполагается, что набор данных находится в `/kaggle/input/sxuprl/train/train` и `/kaggle/input/sxuprl/test/test`. Если ваши данные находятся в другом месте, вам нужно будет скорректировать эти пути в функции `create_dataset`.

### 3\. Загрузите веса YOLOv8-face (Уже включены при использовании Kaggle Input)

Скрипт использует предварительно обученные веса YOLOv8-face `yolov8n-face.pt`. Код предполагает, что эти веса расположены в `/kaggle/input/face-detection-using-yolov8/yolov8n-face.pt`. Если вы запускаете скрипт вне Kaggle или вам нужно использовать другой путь, убедитесь, что инициализация `face_model` в классе `VideoProcessor` (`self.face_model = YOLO("/path/to/your/yolov8n-face.pt")`) указывает на правильное местоположение вашего файла весов YOLOv8-face. Если вы используете предоставленную среду Kaggle notebook и набор данных, этот шаг, вероятно, уже выполнен.

### 4\. Запустите скрипт

1.  **Перейдите в каталог скрипта**: Откройте терминал и перейдите в каталог, где сохранен ваш скрипт Python.
2.  **Выполните скрипт**: Запустите скрипт с помощью Python:
    ```bash
    python your_script_name.py
    ```
    Замените `your_script_name.py` на фактическое имя вашего файла Python.

### 5\. Обучение и вывод

  * **Прогресс обучения**: Во время обучения вы будете видеть обновления прогресса, включая значения потерь и метрик для каждой эпохи, благодаря `TqdmCallback`.
  * **Сохранение модели**: Лучшая модель (на основе RMSE валидации) будет сохранена в `best_model.keras` благодаря колбэку `ModelCheckpoint`.
  * **Логи TensorBoard**: Логи TensorBoard будут сохранены в каталоге `./logs`. Вы можете визуализировать прогресс обучения и графики модели, запустив TensorBoard в своем терминале: `tensorboard --logdir ./logs`.
  * **Финальная оценка**: После обучения скрипт загрузит лучшую сохраненную модель (`best_model.keras`) и оценит ее на тестовом наборе данных. Итоговые тестовые потери и метрики будут выведены в консоль.

## Архитектура модели

Модель глубокого обучения представляет собой сеть 3D CNN-LSTM, разработанную для пространственно-временного извлечения признаков из видеоданных. Вот послойное описание архитектуры:

1.  **Входной слой**: `Input(shape=(WINDOW_SIZE, HEIGHT, WIDTH, 3))`. Принимает входные видеокадры окнами размера `WINDOW_SIZE`. Каждый кадр имеет размеры `HEIGHT` x `WIDTH` с 3 цветовыми каналами (RGB).

2.  **3D Сверточные слои**:

      * `Conv3D(32, (3, 5, 5), padding="same")`: Первый 3D сверточный слой с 32 фильтрами, размером ядра (3, 5, 5) (временной, высота, ширина) и паддингом 'same' для сохранения пространственных размеров.

      * `BatchNormalization()`: Batch normalization для стабилизации обучения и улучшения обобщения.

      * `ReLU()`: Функция активации ReLU для нелинейности.

      * `MaxPool3D((1, 2, 2))`: 3D слой макс-пулинга с размером пула (1, 2, 2) для уменьшения пространственных размеров (высоты и ширины), сохраняя при этом временной размер неизменным.

      * `Conv3D(64, (3, 3, 3), padding="same")`: Второй 3D сверточный слой с 64 фильтрами и размером ядра (3, 3, 3).

      * `BatchNormalization()`, `ReLU()`, `MaxPool3D((1, 2, 2))`: Аналогичные слои, как и в первом блоке, дополнительно извлекающие признаки и уменьшающие пространственный размер.

      * `Conv3D(128, (3, 3, 3), padding="same")`: Третий 3D сверточный слой со 128 фильтрами и размером ядра (3, 3, 3).

      * `BatchNormalization()`, `ReLU()`, `MaxPool3D((1, 2, 2))`: Заключительный 3D сверточный блок.

    3D CNN слои отвечают за извлечение пространственно-временных признаков непосредственно из видеокадров, учитывая движение и изменения внешнего вида во временном окне `WINDOW_SIZE`.

3.  **Слой Reshape**: `Reshape((WINDOW_SIZE, -1))`: Изменяет форму выхода 3D CNN на 2D тензор формы `(WINDOW_SIZE, features)`. Это подготавливает выходные данные для подачи в слои LSTM, сглаживая пространственные карты признаков в один вектор признаков для каждого временного шага в окне.

4.  **Слои LSTM**:

      * `LSTM(128, return_sequences=True)`: Первый слой LSTM со 128 юнитами. `return_sequences=True` гарантирует, что LSTM выводит последовательности для каждого временного шага, что необходимо для последующего слоя LSTM и слоев TimeDistributed.
      * `LSTM(64, return_sequences=True)`: Второй слой LSTM с 64 юнитами, также с `return_sequences=True`.

    Слои LSTM имеют решающее значение для моделирования временных зависимостей в последовательности признаков, извлеченных 3D CNN. Они изучают закономерности и взаимосвязи между кадрами внутри каждого окна.

5.  **Слои TimeDistributed**:

      * `TimeDistributed(layers.Dense(256, activation="relu"))`: Слой TimeDistributed Dense с 256 юнитами и активацией ReLU. Применяется к каждому временному шагу выхода LSTM независимо.
      * `TimeDistributed(layers.Dropout(0.3))`: Слой TimeDistributed Dropout с коэффициентом dropout 0.3, применяется для регуляризации для предотвращения переобучения.
      * `TimeDistributed(layers.Dense(1))`: Заключительный слой TimeDistributed Dense с 1 юнитом, производящий одно выходное значение для каждого временного шага.

    Слои TimeDistributed применяют одни и те же операции Dense и Dropout к каждому временному шагу последовательности, выводимой LSTM. Это позволяет обрабатывать временные признаки для каждого кадра. Заключительный слой Dense с 1 юнитом выполняет задачу регрессии, предсказывая одно непрерывное значение для каждого кадра в окне.

6.  **Выходной слой**: `Reshape((WINDOW_SIZE,))(outputs)`: Изменяет форму выхода из `(WINDOW_SIZE, 1)` в `(WINDOW_SIZE,)`, предоставляя 1D тензор предсказаний для каждого кадра во входном окне.

**Компиляция модели**:

  * **Оптимизатор**: Оптимизатор Adam с расписанием экспоненциального затухания скорости обучения. Скорость обучения начинается с `3e-5` и со временем затухает.
  * **Функция потерь**: Средняя абсолютная ошибка (MAE) используется в качестве функции потерь, подходящей для задач регрессии.
  * **Метрики**: MAE и среднеквадратическая ошибка (RMSE) используются в качестве метрик оценки.

## Процесс обучения

Процесс обучения модели настроен следующим образом:

  * **Создание набора данных**: Функция `create_dataset` настраивает обучающий и тестовый наборы данных, используя `tf.data.Dataset` для эффективной загрузки данных и параллельной обработки.
  * **Распределенное обучение**: Код использует `tf.distribute.MirroredStrategy` для включения распределенного обучения на нескольких GPU, если они доступны. Это может значительно ускорить время обучения.
  * **Коллбэки**: Во время обучения используется несколько колбэков:
      * **`TqdmCallback`**: Обеспечивает визуальный индикатор прогресса для каждой эпохи.
      * **`ModelCheckpoint`**: Сохраняет модель с наилучшей среднеквадратической ошибкой (RMSE) валидации в `best_model.keras`.
      * **`TensorBoard`**: Записывает логи для визуализации в TensorBoard, позволяя отслеживать прогресс обучения, потери и метрики.
      * **`EarlyStopping`**: Завершает обучение раньше, если RMSE валидации не улучшается в течение 3-х последовательных эпох, предотвращая переобучение и экономя время обучения. Он также восстанавливает лучшие веса модели, сохраненные `ModelCheckpoint`.
  * **Эпохи**: Модель обучается максимум 5 эпох, как определено в `epochs=5` в `model.fit`. Возможно, вам потребуется увеличить количество эпох для лучшей сходимости в зависимости от размера и сложности вашего набора данных.
  * **Валидация**: Аргумент `validation_data=test_ds` в `model.fit` указывает, что набор данных `test_ds` будет использоваться для валидации во время обучения.

## Оценка

Оценка модели выполняется после обучения с использованием лучшей сохраненной модели (`best_model.keras`). Функция `best_model.evaluate(test_ds)` вычисляет потери и метрики (MAE и RMSE) на тестовом наборе данных. Затем скрипт выводит окончательную тестовую производительность, которая включает значение тестовых потерь и RMSE.

Основной метрикой для оценки производительности является среднеквадратическая ошибка (RMSE), поскольку она также используется для сохранения модели и ранней остановки. Более низкие значения RMSE указывают на лучшую производительность модели, что означает, что предсказания модели ближе к целевым значениям ground truth.

## Настройка

Этот проект разработан для настройки:

  * **Гиперпараметры**: Вы можете легко настроить различные гиперпараметры:
      * `WINDOW_SIZE`: Измените глобальную переменную `WINDOW_SIZE`, чтобы изменить размер временного входного окна.
      * `HEIGHT`, `WIDTH`: Измените `HEIGHT` и `WIDTH`, чтобы изменить размер кадров до других размеров.
      * `YOLO_BATCH_SIZE`, `DATA_BATCH_SIZE`: Настройте размеры пакетов в зависимости от возможностей вашего оборудования и размера набора данных.
      * Расписание скорости обучения: Измените `lr_schedule` в `build_3dcnn_lstm_model`, чтобы поэкспериментировать с различными стратегиями затухания скорости обучения или начальными скоростями обучения.
      * Архитектура модели: Функция `build_3dcnn_lstm_model` может быть изменена для экспериментов с различными конфигурациями слоев:
          * Количество 3D CNN слоев и фильтров.
          * Размеры ядер в слоях CNN.
          * Количество слоев и юнитов LSTM.
          * Функции активации.
          * Методы регуляризации (Dropout, Batch Normalization).
  * **Пути к набору данных**: Измените пути `data_dir` в функции `create_dataset`, чтобы указать на пользовательские местоположения набора данных.
  * **Модель обнаружения лиц**: Вы можете заменить модель YOLOv8-face другой моделью обнаружения лиц, если необходимо, изменив строку `face_model = YOLO(...)` в классе `VideoProcessor`.
  * **Параметры обучения**: Настройте количество эпох, колбэки и настройки оптимизатора в основном разделе скрипта.

## Потенциальные улучшения

  * **Аугментация данных**: Внедрение методов аугментации данных для видеоданных (например, временной джиттеринг, небольшие повороты, масштабирование, цветовые аугментации в пределах области лица) может улучшить устойчивость и обобщение модели.
  * **Продвинутые архитектуры моделей**: Изучите более продвинутые архитектуры, такие как:
      * Модели на основе Transformer для моделирования временных последовательностей.
      * Использование различных бэкбонов CNN (например, бэкбоны ResNet, EfficientNet, адаптированные для 3D CNN).
      * Механизмы внимания внутри слоев LSTM или CNN, чтобы сосредоточиться на более релевантных признаках.
  * **Тонкая настройка YOLOv8-face**: Если точность обнаружения лиц критична, а модель YOLOv8-face не идеально подходит для вашего набора данных, рассмотрите возможность ее тонкой настройки на подмножестве ваших данных или использования детектора лиц, специально обученного для русских лиц, если таковой имеется.
  * **Эксперименты с функцией потерь**: Поэкспериментируйте с различными функциями потерь, помимо MAE, такими как потеря Хубера или квантильная потеря, в зависимости от желаемых свойств предсказаний.
  * **Больший набор данных**: Обучение на большем и более разнообразном наборе данных видео на русском языке, вероятно, приведет к улучшению производительности и обобщения модели.

## Лицензия

Этот проект выпущен под [лицензией MIT](https://www.google.com/url?sa=E&source=gmail&q=https://www.google.com/url?sa=E%26source=gmail%26q=LICENSE) (или укажите выбранную вами лицензию).

