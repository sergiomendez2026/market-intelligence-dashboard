Arquitectura

1. Visión general

Market Intelligence Dashboard sigue una arquitectura modular diseñada para separar responsabilidades entre carga de datos, cálculo de indicadores, ingeniería de variables, entrenamiento de modelos, generación de señales, validación, backtesting, visualización e interfaz de usuario.

La aplicación está desplegada con Streamlit y organizada como producto analítico, no como un script monolítico.

---

2. Estructura del repositorio

market-intelligence-dashboard/
│
├── app.py
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── indicators.py
│   ├── features.py
│   ├── model.py
│   ├── model_comparison.py
│   ├── signals.py
│   ├── sentiment.py
│   ├── financial_metrics.py
│   ├── backtesting.py
│   ├── validation.py
│   ├── visualization.py
│   └── ui.py
│
├── tests/
│   ├── __init__.py
│   ├── test_indicators.py
│   ├── test_signals.py
│   └── test_financial_metrics.py
│
├── notebooks/
│   ├── README.md
│   └── 01_eda_market_data.ipynb
│
├── reports/
│   ├── methodology.md
│   └── model_card.md
│
├── docs/
│   ├── metodologia.md
│   ├── arquitectura.md
│   └── roadmap.md
│
├── assets/
│   └── .gitkeep
│
└── .github/
    └── workflows/
        └── python-tests.yml

---

3. Responsabilidad por módulo

Módulo| Responsabilidad
"app.py"| Aplicación principal en Streamlit y orquestación general.
"src/config.py"| Constantes, parámetros y configuración global.
"src/data_loader.py"| Carga de datos financieros históricos.
"src/indicators.py"| Cálculo de indicadores técnicos.
"src/features.py"| Construcción del dataset de machine learning.
"src/model.py"| Entrenamiento y evaluación de modelos ML.
"src/model_comparison.py"| Comparación entre modelos supervisados.
"src/signals.py"| Construcción del Market Signal Score.
"src/sentiment.py"| Sentimiento financiero con fallback liviano.
"src/financial_metrics.py"| KPIs financieros: retorno, volatilidad y drawdown.
"src/backtesting.py"| Simulación de estrategia y métricas de backtesting.
"src/validation.py"| Validación walk-forward para series temporales.
"src/visualization.py"| Visualizaciones reutilizables con Plotly.
"src/ui.py"| Componentes visuales y layout de Streamlit.

---

4. Flujo de datos

Datos de mercado
    ↓
Carga de datos
    ↓
Indicadores técnicos
    ↓
Ingeniería de variables
    ↓
Dataset de machine learning
    ↓
Entrenamiento y comparación de modelos
    ↓
Generación de señal de mercado
    ↓
Backtesting y validación temporal
    ↓
Dashboard en Streamlit

---

5. Capas de la aplicación

5.1 Capa de datos

Responsable de obtener y preparar datos históricos de mercado.

Implementación actual:

- "yfinance".
- Precios históricos.
- Selección por ticker.
- Cache mediante Streamlit.

5.2 Capa de indicadores

Responsable de calcular variables técnicas derivadas de los precios.

Incluye:

- Media móvil de 20 períodos.
- Media móvil de 50 períodos.
- Bandas de Bollinger.
- RSI.
- Volatilidad móvil.
- Retornos.
- Momentum.

5.3 Capa de ingeniería de variables

Responsable de convertir precios e indicadores en variables predictivas.

Incluye:

- Retornos rezagados.
- Volatilidad histórica.
- Momentum.
- Distancia frente a medias móviles.
- Variables de tendencia.
- Objetivo de regresión.
- Objetivo direccional.

5.4 Capa de modelado

Responsable de entrenar y evaluar modelos supervisados.

Incluye:

- Modelo de regresión.
- Modelo direccional.
- Comparación contra baseline.
- Comparación entre modelos.
- Robustness Score.

5.5 Capa de validación

Responsable de evaluar modelos bajo una lógica temporal más rigurosa.

Incluye:

- Validación walk-forward.
- Evaluación en ventanas futuras.
- Comparación contra baseline.
- Métricas direccionales.

5.6 Capa de señales

Responsable de combinar probabilidad del modelo, análisis técnico, sentimiento y volatilidad en una señal ejecutiva.

Componentes:

- Probabilidad alcista ML.
- Score técnico.
- Score de sentimiento.
- Score de volatilidad.

5.7 Capa de backtesting

Responsable de simular el comportamiento de una estrategia bajo supuestos explícitos.

Incluye:

- Benchmark Buy & Hold.
- Comisiones.
- Slippage.
- Exposición.
- Retorno de estrategia.
- Drawdown.
- Sharpe Ratio aproximado.

5.8 Capa de interfaz

Responsable de presentar resultados en un dashboard ejecutivo.

Incluye:

- Sidebar.
- Tabs.
- Métricas.
- Gráficos.
- Alertas.
- Notas metodológicas.
- Disclaimer.

---

6. Arquitectura de despliegue

Repositorio GitHub
      ↓
Streamlit Cloud
      ↓
Dashboard interactivo

GitHub Pages
      ↓
Página web del proyecto

GitHub Actions
      ↓
Testing automatizado

Superficies públicas del proyecto:

Superficie| Propósito
Repositorio GitHub| Código fuente, documentación, tests y control de versiones.
Página GitHub Pages| Presentación pública tipo portafolio.
Dashboard Streamlit| Producto analítico interactivo.
GitHub Actions| Validación automática de tests.
Releases| Control formal de versiones.

---

7. Control de calidad

Controles actuales:

- Tests unitarios.
- GitHub Actions.
- Dependencias de desarrollo.
- Arquitectura modular.
- Changelog.
- Licencia MIT.
- Documentación técnica.
- Validación local en Google Colab.
- Validación cloud en GitHub Actions.

Áreas testeadas actualmente:

- Indicadores técnicos.
- Métricas financieras.
- Señales de mercado.

Mejoras futuras:

- Mayor cobertura de tests.
- Tests de integración.
- Tests de regresión.
- Validación de datos.
- Smoke tests de interfaz.
- Tests de estabilidad de modelos.
- Pruebas de rendimiento.

---

8. Principios de diseño

La arquitectura sigue los siguientes principios:

- Separación de responsabilidades.
- Modularidad.
- Reproducibilidad.
- Extensibilidad.
- Interpretabilidad.
- Validación contra baseline.
- Transparencia metodológica.
- Diseño orientado a producto analítico.

---

9. Escalabilidad futura

La arquitectura permite evolucionar hacia una plataforma más robusta.

Extensiones potenciales:

- API con FastAPI.
- Base de datos PostgreSQL.
- Autenticación de usuarios.
- Sistema multiusuario.
- Reportes automáticos en PDF.
- Alertas por email, Telegram o WhatsApp.
- Integración de FinBERT.
- Optimización de portafolio.
- Frontend profesional con React o Next.js.
- Despliegue con Docker.
- CI/CD más avanzado.

---

10. Disclaimer

Este proyecto tiene fines educativos, analíticos y demostrativos. No constituye asesoría financiera, recomendación de inversión ni promesa de rentabilidad.
