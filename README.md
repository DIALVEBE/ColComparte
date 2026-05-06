# Colombia Comparte RAG Chatbot

Chatbot RAG para responder preguntas sobre Colombia Comparte con contexto real de los documentos del proyecto.

Flujo principal:

```text
Usuario -> Embedding -> FAISS/coseno -> Contexto -> Qwen2.5 -> Respuesta
```

El objetivo es responder con naturalidad sin inventar información. Si el contexto recuperado no alcanza, el chatbot usa fallback:

```text
No tengo suficiente información para responder esa pregunta con los datos disponibles.
```

## Enfoque

El chunking no corta por caracteres. La estrategia es:

1. Leer documentos desde `data/raw`.
2. Limpiar espacios sin perder los saltos entre parrafos.
3. Quitar ruido editorial de documentos de trabajo, como instrucciones de diseno, CTAs internos, notas pendientes y texto de revision.
4. Detectar fronteras semanticas como secciones, slides, rutas, programas y titulos numerados.
5. Agrupar parrafos completos hasta llegar a un tamano objetivo, respetando esas fronteras.
6. Si un parrafo extraido de PDF es demasiado largo, dividirlo primero por estructura interna y luego por oraciones.
7. Agregar una oracion de solapamiento solo cuando el corte no sea cambio de tema.
8. Fusionar chunks demasiado cortos o que terminen en una idea abierta.
9. Validar cada chunk para detectar finales raros, ideas cortadas o tamanos fuera de rango.
10. Generar embeddings con un modelo pequeno de Hugging Face.
11. Guardar `chunks.jsonl`, `embeddings.npy` y un indice FAISS.

Modelo recomendado para embeddings:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Modelo de generación recomendado por el reto:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

## Estructura

```text
data/raw/                 documentos originales
data/processed/           chunks y embeddings generados
indexes/                  indice FAISS
scripts/build_knowledge_base.py
scripts/search_semantic.py
scripts/chatbot.py
streamlit_app.py
src/rag/                  modulos reutilizables
```

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Preparar documentos

Coloca los archivos de Colombia Comparte en:

```text
data/raw/
```

Formatos soportados: `.txt`, `.md`, `.docx`, `.pdf`.

## Construir la base de conocimiento

```bash
python scripts/build_knowledge_base.py
```

Parametros utiles:

```bash
python scripts/build_knowledge_base.py --min-words 35 --target-words 140 --max-words 240
```

Para revisar solo la calidad de los chunks sin regenerar embeddings:

```bash
python scripts/build_knowledge_base.py --skip-embeddings
```

Si el modelo de embeddings no esta en cache local, permite la descarga con:

```bash
python scripts/build_knowledge_base.py --download-model
```

Archivos generados:

```text
data/processed/chunks.jsonl
data/processed/embeddings.npy
indexes/faiss.index
```

## Probar busqueda semantica

```bash
python scripts/search_semantic.py "Que hace Colombia Comparte?"
```

El script muestra los chunks completos mas similares junto con su score. Si quieres una salida corta para inspeccion rapida, usa `--preview-chars 450`.

## Probar chatbot con Qwen

```bash
python scripts/chatbot.py "Que es EDIFICA?"
```

Opciones utiles:

```bash
python scripts/chatbot.py "Que es EDIFICA?" --show-context
python scripts/chatbot.py "Que es EDIFICA?" --no-generate
python scripts/chatbot.py "Que es EDIFICA?" --top-k 4 --min-score 0.25
```

`--no-generate` sirve para probar solo retrieval sin cargar Qwen.
Si Qwen o el modelo de embeddings no estan en cache local, usa `--download-models`.

## Interfaz grafica

La interfaz grafica esta hecha con Streamlit como una landing sencilla con chat integrado:

```bash
streamlit run streamlit_app.py
```

Abre:

```text
http://localhost:8501
```

La experiencia visual oculta la configuracion tecnica para que el usuario vea una pagina sencilla:

```text
Landing de Colombia Comparte
Boton para abrir el chat
Preguntas sugeridas
Panel conversacional
Respuestas generadas con RAG + Qwen
```

La app carga `data/processed/chunks.jsonl` e `indexes/faiss.index`. Si esos archivos no existen, ejecuta primero:

```bash
python scripts/build_knowledge_base.py
```

## Validacion de chunks

Cada registro en `chunks.jsonl` incluye:

```json
{
  "id": "chunk-0001",
  "text": "...",
  "source": "data/raw/documento.docx",
  "word_count": 320,
  "validation_status": "ok",
  "validation_notes": []
}
```

Si `validation_status` queda en `review`, el chunk no se descarta: queda marcado para revision porque puede estar muy corto, muy largo o terminar con una idea aparentemente incompleta.
