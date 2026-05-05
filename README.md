# Colombia Comparte RAG

Base inicial para construir el chatbot RAG del proyecto. Esta primera parte cubre la semana 1: carga de documentos, limpieza, chunks de buena calidad, embeddings y busqueda semantica con FAISS.

## Enfoque de la semana 1

El chunking no corta por caracteres. La estrategia es:

1. Leer documentos desde `data/raw`.
2. Limpiar espacios sin perder los saltos entre parrafos.
3. Agrupar parrafos completos hasta llegar a un tamano objetivo.
4. Si un parrafo es demasiado largo, dividirlo por oraciones.
5. Agregar una oracion de solapamiento entre chunks para mantener contexto.
6. Validar cada chunk para detectar finales raros, ideas cortadas o tamanos fuera de rango.
7. Opcionalmente revisar chunks dudosos con un modelo pequeno de Hugging Face.
8. Generar embeddings con un modelo pequeno de Hugging Face.
9. Guardar `chunks.jsonl`, `embeddings.npy` y un indice FAISS.

Modelo recomendado para embeddings:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Es liviano, funciona en espanol y se descarga desde Hugging Face.

## Estructura

```text
data/raw/                 documentos originales
data/processed/           chunks y embeddings generados
indexes/                  indice FAISS
scripts/build_knowledge_base.py
scripts/search_semantic.py
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
python scripts/build_knowledge_base.py --min-words 180 --target-words 320 --max-words 500
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

El script muestra los chunks mas similares junto con su score. Esta prueba sirve para revisar si los fragmentos recuperados tienen sentido antes de integrar el LLM.

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

## Apoyo opcional con LLM pequeno

Para semana 1 conviene que el chunking principal use reglas claras y auditables. Aun asi, el proyecto incluye `src/rag/chunk_reviewer.py`, que permite revisar manualmente un chunk dudoso con `google/flan-t5-small` desde Hugging Face.

Uso esperado:

```python
from rag.chunk_reviewer import review_chunk_with_small_llm

decision = review_chunk_with_small_llm(chunk_text)
print(decision)  # OK o REVIEW
```

No se ejecuta por defecto porque descargar y cargar un LLM hace mas lento el pipeline. La recomendacion es usarlo solo sobre chunks con `validation_status = "review"`.
