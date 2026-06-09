import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Определение признаков V1 (глобальная переменная для использования в других функциях)
V1_COLUMNS = ['N1', 'N2', 'N3', 'V1real', 'V2real', 'V3real', 
              'I1', 'I2', 'I3', 'gx', 'gy', 'gz', 'ax', 'ay', 'az']

# Определение признаков V2 (глобальная переменная для использования в других функциях)
V2_COLUMNS = [
    'Vx', 'Vy', 'Omega', 'Ix', 'Iy', 'Iphi', 'Isum']


def prepare_data_v1(data, target_type=5):
    """
    Подготовка данных для варианта V1
    
    Parameters:
    -----------
    data : DataFrame
        Исходные данные
    target_type : int
        Тип поверхности для классификации (по умолчанию 1)
        
    Returns:
    --------
    X : DataFrame
        Признаки V1
    y : Series
        Бинарная целевая переменная
    """
    # Создаем бинарную целевую переменную
    y = (data['Type'] == target_type).astype(int)
    
    # Берем все признаки V1
    X = data[V1_COLUMNS]
    
    return X, y

def calculate_v2_features(data):
    """
    Вычисление интегральных признаков V2.

    V2 = {
        Vx, Vy,
        Omega,
        Ix, Iy,
        Iphi, Isum
    }
    """

    df = data.copy()

    # параметры платформы
    R = 40.0      # мм
    L = 125.0     # мм

    alpha = 0.0
    theta = np.deg2rad(30)

    w1 = df['V1real']
    w2 = df['V2real']
    w3 = df['V3real']

    I1 = df['I1']
    I2 = df['I2']
    I3 = df['I3']

    c1 = np.cos(alpha - theta)
    c2 = np.cos(alpha)
    c3 = np.cos(alpha + theta)

    s1 = np.sin(alpha - theta)
    s2 = np.sin(alpha)
    s3 = np.sin(alpha + theta)

    # ------------------------------------
    # Скорости движения робота
    # ------------------------------------

    df['Vx'] = R * (
        (-2/3) * c1 * w1 +
        ( 2/3) * s2 * w2 +
        ( 2/3) * c3 * w3
    )

    df['Vy'] = R * (
        (-2/3) * s1 * w1 +
        (-2/3) * c2 * w2 +
        ( 2/3) * s3 * w3
    )

    df['Omega'] = R * (
        (1/(3*L))*w1 +
        (1/(3*L))*w2 +
        (1/(3*L))*w3
    )

    # ------------------------------------
    # Трудоемкость движения
    # ------------------------------------

    df['Ix'] = (
        (-2/3) * c1 * I1 +
        ( 2/3) * s2 * I2 +
        ( 2/3) * c3 * I3
    )

    df['Iy'] = (
        (-2/3) * s1 * I1 +
        (-2/3) * c2 * I2 +
        ( 2/3) * s3 * I3
    )

    df['Iphi'] = (
        (1/3) * I1 +
        (1/3) * I2 +
        (1/3) * I3
    )

    df['Isum'] = np.sqrt(
        df['Ix']**2 +
        df['Iy']**2
    )

    return df

def prepare_data_v2(data, target_type=5):
    """
    Подготовка данных для варианта V2

    Parameters:
    -----------
    data : DataFrame
        Исходные данные (уже содержащие признаки V2)
    target_type : int
        Тип поверхности для классификации

    Returns:
    --------
    X : DataFrame
        Признаки V2
    y : Series
        Бинарная целевая переменная
    """

    y = (data['Type'] == target_type).astype(int)

    X = data[V2_COLUMNS]

    return X, y

def prepare_data_v1v2(data, target_type=5):
    """
    Подготовка данных для варианта V1 + V2

    Parameters:
    -----------
    data : DataFrame
        Исходные данные (уже содержащие признаки V2)
    target_type : int
        Тип поверхности для классификации

    Returns:
    --------
    X : DataFrame
        Совмещенные признаки V1 и V2
    y : Series
        Бинарная целевая переменная
    """

    y = (data['Type'] == target_type).astype(int)

    X = data[V1_COLUMNS + V2_COLUMNS]

    return X, y

def evaluate_model(X, y, model, cv=3):
    """
    Оценка модели с кросс-валидацией
    
    Parameters:
    -----------
    X : DataFrame/array
        Признаки
    y : Series/array
        Целевая переменная
    model : sklearn model
        Модель для оценки
    cv : int
        Число фолдов для кросс-валидации
        
    Returns:
    --------
    results : dict
        Результаты оценки
    """
    # Кросс-валидация
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    # Обучение на всех данных
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Метрики
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    return {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores,
        'accuracy': accuracy,
        'f1': f1,
        'y_pred': y_pred,
        'model': model
    }


def plot_training_results(y_true, y_pred, title="Результаты обучения", model_info=""):
    """
    Визуализация результатов классификации на обучающих данных
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. График сравнения истинных и предсказанных значений
    sample_size = min(80, len(y_true))
    indices = np.arange(sample_size)
    
    axes[0].plot(indices, y_true[:sample_size], 'o-', label='Истинные', 
                 alpha=0.7, markersize=6, linewidth=1, color='blue')
    axes[0].plot(indices, y_pred[:sample_size], 'x-', label='Предсказанные', 
                 alpha=0.7, markersize=6, linewidth=1, color='red')
    axes[0].set_xlabel('Номер образца')
    axes[0].set_ylabel('Класс (0/1)')
    axes[0].set_title('Сравнение истинных и предсказанных значений')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_xlabel('Предсказанный класс')
    axes[1].set_ylabel('Истинный класс')
    axes[1].set_title('Матрица ошибок')
    
    # 3. Отчет классификации
    report = classification_report(y_true, y_pred, target_names=['Тип ≠ 5', 'Тип = 5'], output_dict=True)
    axes[2].axis('off')
    report_text = f"Accuracy: {report['accuracy']:.4f}\n\n"
    report_text += f"Класс 'Тип ≠ 5':\n"
    report_text += f"  Precision: {report['Тип ≠ 5']['precision']:.4f}\n"
    report_text += f"  Recall: {report['Тип ≠ 5']['recall']:.4f}\n"
    report_text += f"  F1: {report['Тип ≠ 5']['f1-score']:.4f}\n\n"
    report_text += f"Класс 'Тип = 5':\n"
    report_text += f"  Precision: {report['Тип = 5']['precision']:.4f}\n"
    report_text += f"  Recall: {report['Тип = 5']['recall']:.4f}\n"
    report_text += f"  F1: {report['Тип = 5']['f1-score']:.4f}"
    
    axes[2].text(0.5, 0.5, report_text, ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))
    axes[2].set_title('Отчет классификации')
    
    plt.suptitle(f"{title}\n{model_info}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_test_results(y_true, y_pred, train_metrics, test_metrics, title="Результаты на тестовых данных"):
    """
    Визуализация результатов на тестовых данных с сравнением с обучающими
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. График сравнения на тестовых данных
    sample_size = min(50, len(y_true))
    indices = np.arange(sample_size)
    
    axes[0].plot(indices, y_true[:sample_size], 'o-', label='Истинные', 
                 alpha=0.7, markersize=6, linewidth=1, color='blue')
    axes[0].plot(indices, y_pred[:sample_size], 'x-', label='Предсказанные', 
                 alpha=0.7, markersize=6, linewidth=1, color='red')
    axes[0].set_xlabel('Номер образца')
    axes[0].set_ylabel('Класс (0/1)')
    axes[0].set_title('Тестовые данные: сравнение')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Матрица ошибок на тестовых данных
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_xlabel('Предсказанный класс')
    axes[1].set_ylabel('Истинный класс')
    axes[1].set_title('Матрица ошибок (тест)')
    
    # 3. Сравнение метрик на тренировочных и тестовых данных
    metrics_comparison = pd.DataFrame({
        'Метрика': ['Accuracy', 'F1-score'],
        'Тренировочные': [train_metrics['accuracy'], train_metrics['f1']],
        'Тестовые': [test_metrics['accuracy'], test_metrics['f1']]
    })
    
    x = np.arange(len(metrics_comparison))
    width = 0.35
    
    bars1 = axes[2].bar(x - width/2, metrics_comparison['Тренировочные'], width, 
                       label='Тренировочные', color='green', alpha=0.7)
    bars2 = axes[2].bar(x + width/2, metrics_comparison['Тестовые'], width, 
                       label='Тестовые', color='orange', alpha=0.7)
    
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(metrics_comparison['Метрика'])
    axes[2].set_ylabel('Значение')
    axes[2].set_title('Сравнение метрик')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()