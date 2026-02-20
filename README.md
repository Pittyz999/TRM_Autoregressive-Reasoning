# Tiny Recursive Model (TRM) - Autoregressive Reasoning
Adaptación del modelo Tiny Recursive Model (TRM) para razonamiento autorregresivo.

### Tiny Recursive Model (TRM) - Autoregressive Math Reasoning
Primera versión del modelo TRM adaptado con razonamiento matemático autorregresivo ejecutado en un entorno de Google Colab.

### Descripción del Proyecto
Este proyecto adapta la arquitectura experimental Tiny Recursive Model (TRM) (originalmente diseñada para resolución de problemas de estado fijo como Sudokus y ARC-AGI) y la transforma en un Modelo de Lenguaje Autorregresivo. El objetivo es dotar al modelo de la capacidad de resolver problemas matemáticos en lenguaje natural (Dataset GSM8K) utilizando recursividad latente para emular "tiempo de pensamiento" (Chain of Thought interno) antes de emitir cada token.

### Logros y Hitos Técnicos Alcanzados
1. Transformación Arquitectónica (De Matriz a Texto)
    * Atención Causal: Modificación de las capas internas del TRM para reemplazar el procesamiento bidireccional/MLP con Auto-Atención Causal (causal=True), permitiendo la generación de texto de izquierda a derecha.
    * Integración de Tokenizer: Acoplamiento exitoso del tokenizador de GPT-2 para procesar lenguaje natural en lugar de enteros fijos.
    * Memoria Dinámica (Carry State): Rediseño del estado latente (z_H, z_L) para que se instancie dinámicamente según la longitud de la secuencia actual, evitando problemas de dimensionalidad de tensores.

2. Correcciones de Bajo Nivel y Optimización
    * RoPE Dinámico (Rotary Positional Embeddings): Implementación de un parche (Monkey Patch) en la función apply_rotary_pos_emb para recortar dinámicamente las matrices de senos y cosenos, permitiendo al modelo procesar secuencias de longitud variable (desde prompts cortos de 7 tokens hasta textos de 512).
    * Sincronización de Dispositivos: Resolución de conflictos de ejecución en PyTorch moviendo explícitamente los estados iniciales de memoria (creados por defecto en CPU) a la memoria CUDA.

3. Sistema de Entrenamiento Robusto
    * Smart Masking (Loss Optimization): Modificación del DataLoader para aislar el token <|endoftext|>. Se implementó un enmascaramiento con índice -100 para los tokens de padding subsecuentes, enseñando al modelo a detener la generación correctamente (evitando el colapso de "puntos infinitos").
    * Checkpointing y Resumability: Creación de un sistema de guardado continuo en Google Drive que almacena no solo los pesos del modelo, sino el estado del optimizador (AdamW), el Scheduler y la Configuración de la Arquitectura. Esto permite pausar y reanudar el entrenamiento en Colab sin pérdida de progreso ni explosión de gradientes.
    * Human-in-the-Loop Security: Implementación de una interfaz gráfica (vía @param en Colab) con una función safety_check que compara la configuración actual con la guardada en Drive, alertando al usuario de discrepancias para evitar corromper el entrenamiento.
    * Fijación de Origen de Datos: Transición al repositorio oficial "openai/gsm8k" de HuggingFace para garantizar la disponibilidad a largo plazo del dataset.

4. Interfaz de Inferencia Aislada
    * Creación de una consola de pruebas que reconstruye la arquitectura exacta del modelo basándose en el diccionario de configuración guardado en el archivo .pt, permitiendo pruebas Zero-Shot con control de temperatura (Temperature) y longitud máxima (Max_Tokens).

### Estado Actual: Época 50
* Pérdida (Loss) alcanzada: 0.1016
* Diagnóstico actual: El modelo ha completado el ciclo de aprendizaje mecánico y ha demostrado que la propagación hacia atrás (backpropagation) a través del tiempo y los ciclos recursivos funciona perfectamente sin desbordar la VRAM de una GPU T4 (15GB).
* Análisis de Generalización: Debido a la gran capacidad asignada (hidden_size=1024) frente a un dataset pequeño (GSM8K), el modelo actual presenta un sobreajuste (Overfitting) fotográfico. Ha aprendido la estructura y formato de las respuestas matemáticas, pero memorizó los valores en lugar de generalizar la lógica aritmética.

### Próximos Pasos (Roadmap)
* Reducción de Capacidad de Memorización: Disminuir el hidden_size (ej. 512) para forzar al modelo a buscar patrones lógicos en lugar de memorizar datos.
* Aumento de Profundidad de Pensamiento: Incrementar los ciclos recursivos latentes (L_cycles=6 o 8) para mejorar la emulación del cálculo matemático interno.
* Early Stopping: Implementar detención temprana automática para detener el entrenamiento cuando el loss alcance el "punto dulce" de generalización (aprox. 1.2 - 1.5).

## Cómo usar este proyecto
1. Abre el notebook `TRM_Math_Reasoning.ipynb` en Google Colab.
2. Ejecuta la celda de configuración de carpeta y montaje de Google Drive.
3. Ejecuta las dependencias y la preparación del TRM y tokenizador.
4. Para entrenar: Ajusta los parámetros en el formulario "Ejecutar entrenamiento con Seguridad" y dale a Play.
5. Para inferencia: Ve a la sección "Consola de Pruebas", escribe tu problema en inglés y ejecuta.

### Requisitos
Este proyecto fue desarrollado en el entorno de Google Colab, en caso de querer usar el modelo en local estos son las depedencias necesarias:
* PyTorch
* Transformers (HuggingFace)
* Datasets (HuggingFace)
* einops
* tqdm
---
### Sobre el desarrollo y uso de IA
Este proyecto fue desarrollado utilizando un enfoque de *Pair-Programming* con inteligencia artificial (Gemini). Como desarrollador principal, dirigí la lógica de adaptación, el diseño del flujo de trabajo y la toma de decisiones arquitectónicas. La IA actuó como mi asistente para la escritura de código estructural, el *debugging* de tensores en PyTorch y la optimización de las limitaciones de hardware en el entorno de Google Colab.

### Referencias y Agradecimientos
Este proyecto se apoya en el increíble trabajo de la comunidad de IA. Todo el crédito a los creadores originales de las siguientes herramientas y arquitecturas:

* **Arquitectura TRM:** [Código base](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) y conceptos del paper ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871) por Alexia Jolicoeur-Martineau.
* **Dataset GSM8K:** Conjunto de datos de problemas matemáticos de [OpenAI (vía HuggingFace)](https://huggingface.co/datasets/openai/gsm8k).
* **Tokenizador:** Tokenizador BPE pre-entrenado de [GPT-2 (OpenAI / HuggingFace)](https://huggingface.co/openai-community/gpt2).
* **Infraestructura:** Entorno de ejecución gratuito proporcionado por [Google Colab](https://colab.research.google.com/).

### Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
