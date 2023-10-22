# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2022/11/4 13:31

from transformers import AutoTokenizer, AutoConfig,AutoImageProcessor,AutoProcessor,AutoFeatureExtractor, CONFIG_MAPPING, PretrainedConfig

__all__ = [
    'load_tokenizer',
    'load_configure',
    'load_imageprocesser',
    'load_processer',
    'load_feature_extractor',
]

def load_tokenizer(tokenizer_name,
                   model_name_or_path=None,
                   class_name = None,
                   cache_dir="",
                   do_lower_case=None,
                   use_fast_tokenizer=True,
                   model_revision="main",
                   use_auth_token=None,
                   **kwargs):
    tokenizer_kwargs = {
        "cache_dir": cache_dir,
        "revision": model_revision,
        "use_auth_token": True if use_auth_token else None,
        **kwargs
    }
    if do_lower_case is not None:
        tokenizer_kwargs['do_lower_case'] = do_lower_case

    if use_fast_tokenizer is not None:
        tokenizer_kwargs['use_fast'] = use_fast_tokenizer

    if class_name is not None:
        tokenizer = class_name.from_pretrained(tokenizer_name or model_name_or_path, **tokenizer_kwargs)
    elif tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    elif model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return tokenizer

def load_configure(config_name,
                   model_name_or_path=None,
                   class_name = None,
                   cache_dir="",
                   model_revision="main",
                   use_auth_token=None,
                   model_type=None,
                   config_overrides=None,
                   bos_token_id=None,
                   pad_token_id=None,
                   eos_token_id=None,
                   sep_token_id=None,
                   return_dict=False,
                   task_specific_params=None,
                   **kwargs):
    config_kwargs = {
        "cache_dir": cache_dir,
        "revision": model_revision,
        "use_auth_token": True if use_auth_token else None,
        "return_dict": return_dict,
        **kwargs
    }
    tmp_kwargs = {
        "bos_token_id": bos_token_id,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "sep_token_id": sep_token_id,
        "task_specific_params": task_specific_params,
    }
    for k in list(tmp_kwargs.keys()):
        if tmp_kwargs[k] is None:
            tmp_kwargs.pop(k)
    if tmp_kwargs:
        config_kwargs.update(tmp_kwargs)

    if class_name is not None:
        config = class_name.from_pretrained(config_name or model_name_or_path, **config_kwargs)
    elif isinstance(config_name,PretrainedConfig):
        for k,v in config_kwargs.items():
            setattr(config_name,k,v)
        config = config_name

    elif config_name:
        config = AutoConfig.from_pretrained(config_name, **config_kwargs)
    elif model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
    elif model_type:
        config = CONFIG_MAPPING[model_type].from_pretrained(model_name_or_path, **config_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new config_gpt2 from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --config_name."
        )
    if config_overrides is not None:
        config.update_from_string(config_overrides)
    return config


def load_imageprocesser(imageprocesser_name,
                        model_name_or_path=None,
                        class_name = None,
                        cache_dir="",
                        model_revision="main",
                        use_auth_token=None,
                        **kwargs):

    image_kwargs = {
        "cache_dir": cache_dir,
        "revision": model_revision,
        "use_auth_token": True if use_auth_token else None,
        **kwargs
    }

    if class_name is not None:
        image_processer = class_name.from_pretrained(imageprocesser_name or model_name_or_path, **image_kwargs)
    elif imageprocesser_name:
        image_processer = AutoImageProcessor.from_pretrained(imageprocesser_name, **image_kwargs)
    elif model_name_or_path:
        image_processer = AutoImageProcessor.from_pretrained(model_name_or_path, **image_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new imageprocesser from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --imageprocesser_name."
        )
    return image_processer


def load_processer(processer_name,
                   model_name_or_path=None,
                   class_name = None,
                   cache_dir="",
                   model_revision="main",
                   use_auth_token=None,
                   **kwargs):

    image_kwargs = {
        "cache_dir": cache_dir,
        "revision": model_revision,
        "use_auth_token": True if use_auth_token else None,
        **kwargs
    }

    if class_name is not None:
        processer = class_name.from_pretrained(processer_name or model_name_or_path, **image_kwargs)
    elif processer_name:
        processer = AutoProcessor.from_pretrained(processer_name, **image_kwargs)
    elif model_name_or_path:
        processer = AutoProcessor.from_pretrained(model_name_or_path, **image_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new processer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --processer_name."
        )
    return processer

def load_feature_extractor(feature_extractor_name,
                           model_name_or_path=None,
                           class_name = None,
                           cache_dir="",
                           model_revision="main",
                           use_auth_token=None,
                           **kwargs):

    ft_kwargs = {
        "cache_dir": cache_dir,
        "revision": model_revision,
        "use_auth_token": True if use_auth_token else None,
        **kwargs
    }

    if class_name is not None:
        feature_extractor = class_name.from_pretrained(feature_extractor_name or model_name_or_path, **ft_kwargs)
    elif feature_extractor_name:
        feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name, **ft_kwargs)
    elif model_name_or_path:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path, **ft_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new processer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --feature_extractor_name."
        )
    return feature_extractor

