import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from pathlib import Path
import os

from calcutalor import Calculator

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")




@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


JSON_FILE_PATH = Path("function-definitions.json")
@app.get("/get-json", response_class=JSONResponse)
async def get_json():
    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": "JSON file not found"}
        )
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=500,
            content={"error": "Error decoding JSON file"}
        )

@app.post("/calculate/")
async def calculate(startValues=Form(...), maxValues=Form(...), coefs=Form(...), qcoefs=Form(...), normValues=Form(...)):
    startValues_vals = json.loads(startValues)
    maxValues_vals = json.loads(maxValues)
    coefs_vals = json.loads(coefs)
    qcoefs_vals = json.loads(qcoefs)
    normValues_vals = json.loads(normValues)

    calc = Calculator(startValues_vals, maxValues_vals, coefs_vals, qcoefs_vals, normValues_vals)
    time_intervals = np.linspace(0, 1, 11)
    solution = calc.calculate(time_intervals)

    # Разделение решения на отдельные переменные для удобства
    L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14, L15 = solution.T

    # Визуализация графика времени
    fig1, ax1 = plt.subplots(figsize=(16, 8))  # Увеличиваем график

    # Определяем все линии для удобства
    lines = [
        (L1, 'Время испарения'), (L2, 'Время ликвидации'), (L3, 'Площадь заражения'),
        (L4, 'Время подхода облака'), (L5, 'Потери первичного облака'), (L6, 'Потери вторичного облака'),
        (L7, 'Получившие амбулаторную помощь'), (L8, 'Размещенные в стационаре'), (L9, 'Количество поражённой техники'),
        (L10, 'Растворы обеззараживания местности'), (L11, 'Силы и средства для спас. работ'), (L12, 'Эфф. системы оповещения'),
        (L13, 'Людей в зоне поражения'), (L14, 'Спасателей в зоне поражения'), (L15, 'Развитость системы МЧС')
    ]

    # Строим линии графика
    for L, label in lines:
        ax1.plot(time_intervals, L, label=label)

    # Отображаем значения на графике
    for L, label in lines:
        for x, y in zip(time_intervals, L):
            ax1.annotate(f'{y:.2f}', xy=(x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax1.set_xlabel('Время')
    ax1.set_ylabel('Значения')
    ax1.set_title('График времени')

    # Устанавливаем легенду вне графика
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()

    # Сохранение первого графика в буфер
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png")
    buf1.seek(0)
    img_str1 = base64.b64encode(buf1.getvalue()).decode("utf-8")

    # Названия категорий
    categories = [
        "Время испарения", "Время ликвидации", "Площадь заражения", "Время подхода облака", "Потери первичного облака",
        "Потери вторичного облака", "Получившие амбулаторную помощь", "Размещенные в стационаре", "Количество поражённой техники",
        "Растворы обеззараживания местности", "Силы и средства для спас. работ", "Эфф. системы оповещения", "Людей в зоне поражения",
        "Спасателей в зоне поражения", "Развитость системы МЧС"
    ]

    # Углы для категорий
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # Допустим, maxValues — это список из максимальных значений для каждой категории
    maxValues = calc.maxValues
    maxValues += maxValues[:1]

    # Строим графики для каждого времени
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))

    for i, ax in enumerate(axes.flat):
        # Получаем значения для момента времени i
        values = solution[i].tolist()

        # Замыкаем полигон (чтобы соединить последний лепесток с первым)
        values += values[:1]

        # Строим основную диаграмму
        ax.fill(angles, values, color='blue', alpha=0.25)
        ax.plot(angles, values, color='blue', linewidth=2)

        # Добавляем линию для maxValues
        ax.plot(angles, maxValues, color='red', linewidth=2, linestyle='--', label='Max Values')

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_title(f't = {time_intervals[i]}', size=16, y=1.1)

    # Добавляем легенду
    plt.legend(loc='upper right')

    # Устанавливаем плотную компоновку
    plt.tight_layout()

    # Сохранение второго графика в буфер
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png")
    buf2.seek(0)
    img_str2 = base64.b64encode(buf2.getvalue()).decode("utf-8")

    # Закрытие фигур после сохранения
    plt.close(fig1)
    plt.close(fig2)

    return {
        "image1": img_str1,
        "image2": img_str2
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)