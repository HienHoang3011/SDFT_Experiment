"""Qwen2 base-model tokenizer setup for conversational SFT."""

QWEN2_BASE_MODEL_ID = "Qwen/Qwen2-7B"

# This is the template shipped by Qwen/Qwen2-7B. It serializes ChatML but does
# not expose assistant spans to `return_assistant_tokens_mask=True`.
QWEN2_BASE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}"
    "{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + "
    "'<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

# Same rendered ChatML text as the checkpoint template. The Jinja generation
# block only adds metadata: assistant content and <|im_end|> become the loss
# span, while role headers, system text, and user text stay masked.
QWEN2_BASE_TRAINING_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}"
    "{% endif %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% generation %}"
    "{{ message['content'] + '<|im_end|>' + '\n' }}"
    "{% endgeneration %}"
    "{% else %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + "
    "'<|im_end|>' + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def configure_qwen2_base_tokenizer(tokenizer, model_name_or_path):
    """Install a loss-mask-aware ChatML template for Qwen2-7B base."""
    normalized_name = str(model_name_or_path).replace("\\", "/").rstrip("/").lower()
    targets_qwen2_base = normalized_name.endswith("qwen/qwen2-7b") or normalized_name.endswith(
        "/qwen2-7b"
    )
    has_qwen2_base_template = tokenizer.chat_template == QWEN2_BASE_CHAT_TEMPLATE
    already_configured = (
        tokenizer.chat_template is not None
        and "{% generation %}" in tokenizer.chat_template
        and "<|im_start|>" in tokenizer.chat_template
    )

    if not (targets_qwen2_base or has_qwen2_base_template or already_configured):
        return False

    required_tokens = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
    missing_tokens = [token for token in required_tokens if token not in tokenizer.get_vocab()]
    if missing_tokens:
        raise ValueError(
            "Qwen2 base SFT requires the original ChatML special tokens; missing: "
            + ", ".join(missing_tokens)
        )

    if not already_configured:
        if not has_qwen2_base_template:
            raise ValueError(
                "The requested Qwen2-7B tokenizer has an unexpected chat template; "
                "refusing to replace it silently."
            )
        tokenizer.chat_template = QWEN2_BASE_TRAINING_CHAT_TEMPLATE

    # The base checkpoint uses <|endoftext|> as EOS, but ChatML assistant turns
    # end with <|im_end|>. SFT and evaluation must agree on the latter.
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|endoftext|>"
    _validate_assistant_mask(tokenizer)
    return True


def _validate_assistant_mask(tokenizer):
    """Fail before model loading if the installed Transformers cannot mask ChatML."""
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = encoded["input_ids"]
    assistant_mask = encoded.get("assistant_masks")
    if assistant_mask is None or not any(assistant_mask):
        raise RuntimeError(
            "Qwen2 base chat template did not produce an assistant token mask."
        )

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_end_positions = [
        index for index, token_id in enumerate(input_ids) if token_id == im_end_id
    ]
    if not im_end_positions or not assistant_mask[im_end_positions[-1]]:
        raise RuntimeError("The assistant <|im_end|> token is outside the loss mask.")
    if any(assistant_mask[index] for index in im_end_positions[:-1]):
        raise RuntimeError("A system/user <|im_end|> token leaked into the loss mask.")


def sync_model_special_tokens(model, tokenizer):
    """Keep model/generation configs aligned with the configured tokenizer."""
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
