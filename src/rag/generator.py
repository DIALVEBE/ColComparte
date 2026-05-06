from __future__ import annotations

import re


DEFAULT_GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_ANSWER = "No tengo suficiente información para responder esa pregunta con los datos disponibles."


def load_generation_model(model_name: str = DEFAULT_GENERATION_MODEL, local_files_only: bool = True):
    """Load the instruction model used to write final chatbot answers."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("Install torch and transformers to generate chatbot answers.") from error

    # Resolve the Hugging Face id to a local snapshot when possible to avoid network lookups.
    model_path = _resolve_model_path(model_name, local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=local_files_only,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_answer(
    query: str,
    context: str,
    tokenizer,
    model,
    max_new_tokens: int = 220,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """Generate a natural Spanish answer grounded only in retrieved context."""
    if not context.strip():
        return FALLBACK_ANSWER

    # Qwen receives the retrieved chunks as the only allowed source of truth.
    prompt = _build_messages(query, context)
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return _clean_answer(answer)


def _build_messages(query: str, context: str) -> list[dict]:
    # The prompt is intentionally strict because the project evaluates hallucination control.
    system_prompt = (
        "Eres el chatbot oficial de Colombia Comparte. "
        "Responde en español natural, claro y cercano. "
        "Usa únicamente la información del contexto recuperado. "
        "Contesta en máximo tres frases, usando términos que aparezcan en el contexto. "
        "Resume el contexto sin reinterpretarlo, adornarlo ni agregar explicaciones generales. "
        "No inventes datos, cifras, enlaces, programas ni nombres. "
        "No agregues características que no aparezcan explícitamente en el contexto. "
        "Evita palabras genéricas como plataforma, herramienta o servicio si no aparecen en el contexto. "
        "Si el contexto no contiene la respuesta, di exactamente: "
        f"'{FALLBACK_ANSWER}'"
    )
    user_prompt = (
        "Contexto recuperado:\n"
        f"{context}\n\n"
        "Pregunta del usuario:\n"
        f"{query}\n\n"
        "Respuesta:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _clean_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^(Respuesta:|Asistente:)\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip() or FALLBACK_ANSWER


def _resolve_model_path(model_name: str, local_files_only: bool) -> str:
    if "/" not in model_name:
        return model_name

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_name

    return snapshot_download(repo_id=model_name, local_files_only=local_files_only)
