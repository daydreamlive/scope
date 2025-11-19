# Copied from https://github.com/NVlabs/LongLive/blob/main/utils/lora_utils.py
import peft
import torch


def configure_lora_for_model(transformer, model_name, lora_config):
    """Configure LoRA for a WanDiffusionWrapper model

    Args:
        transformer: The transformer model to apply LoRA to
        model_name: 'generator' or 'fake_score'
        lora_config: LoRA configuration
        is_main_process: Whether this is the main process (for logging)

    Returns:
        lora_model: The LoRA-wrapped model
    """
    # Find all Linear modules in WanAttentionBlock modules
    target_linear_modules = set()

    # Define the specific modules we want to apply LoRA to
    if model_name == "generator":
        adapter_target_modules = ["CausalWanAttentionBlock"]
    elif model_name == "fake_score":
        adapter_target_modules = ["WanAttentionBlock"]
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    for name, module in transformer.named_modules():
        if module.__class__.__name__ in adapter_target_modules:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_submodule_name)

    target_linear_modules = list(target_linear_modules)

    # Create LoRA config
    adapter_type = lora_config.get("type", "lora")
    if adapter_type == "lora":
        peft_config = peft.LoraConfig(
            r=lora_config.get("rank", 16),
            lora_alpha=lora_config.get("alpha", None) or lora_config.get("rank", 16),
            lora_dropout=lora_config.get("dropout", 0.0),
            target_modules=target_linear_modules,
        )
    else:
        raise NotImplementedError(f"Adapter type {adapter_type} is not implemented")

    # Apply LoRA to the transformer
    lora_model = peft.get_peft_model(transformer, peft_config)

    return lora_model


def load_lora_checkpoint(model, lora_path: str):
    lora_checkpoint = torch.load(lora_path, map_location="cpu")
    # Support both formats: containing the `generator_lora` key or a raw LoRA state dict
    if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
        peft.set_peft_model_state_dict(model, lora_checkpoint["generator_lora"])
    else:
        peft.set_peft_model_state_dict(model, lora_checkpoint)
