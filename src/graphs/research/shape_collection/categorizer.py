"""
DNN Model Classifier

Classify models into DNN architecture classes based on their names.
"""

from typing import Optional
import re


class DNNClassifier:
    """
    Classify models into DNN architecture classes.

    Classes:
    - CNN: Convolutional Neural Networks (ResNet, VGG, MobileNet, etc.)
    - Encoder: Encoder-only transformers (BERT, ViT, Swin, etc.)
    - Decoder: Decoder-only transformers (GPT-2, OPT, etc.)
    - FullTransformer: Encoder-decoder transformers (T5, BART, etc.)
    """

    # CNN architectures - primarily use convolutions for feature extraction
    CNN_PATTERNS = [
        'resnet', 'resnext', 'wide_resnet',
        'vgg',
        'mobilenet', 'mobilenetv2', 'mobilenetv3',
        'efficientnet', 'efficientnetv2',
        'densenet',
        'shufflenet',
        'squeezenet',
        'regnet',
        'convnext',
        'inception',
        'googlenet',
        'alexnet',
        'nasnet',
        'mnasnet',
        'resnest',
        'res2net',
        'darknet',
        'cspdarknet',
        'yolo',
        'unet',
        'segnet',
        'deeplabv3',
    ]

    # Encoder-only transformers - bidirectional attention, no autoregressive decoding
    ENCODER_PATTERNS = [
        'bert', 'distilbert',
        'roberta',
        'albert',
        'electra',
        'deberta',
        'xlm',
        'xlnet',
        'longformer',
        'bigbird',
        'funnel',
        'reformer',
        'deit',  # Data-efficient Image Transformer
        'beit',  # BERT pre-training of Image Transformers
        'clip',  # CLIP encoder
        # Vision Transformers (encoder-only)
        'vit_', 'vit-',  # Vision Transformer
        'swin_', 'swin-',  # Swin Transformer
        'pvt',  # Pyramid Vision Transformer
        'twins',  # Twins-SVT
        'coat',  # Co-Scale Conv-Attentional Transformers
        'levit',  # LeViT
        'cait',  # Class-Attention in Image Transformers
        'xcit',  # Cross-Covariance Image Transformer
        'poolformer',
        'convmixer',
        'mlp_mixer', 'mlpmixer',
    ]

    # Decoder-only transformers - autoregressive, causal attention
    DECODER_PATTERNS = [
        'gpt', 'gpt2', 'gpt-',
        'opt-', 'opt_',
        'bloom',
        'llama',
        'falcon',
        'mistral',
        'phi-', 'phi_',
        'qwen',
        'codegen',
        'santacoder',
        'starcoder',
        'codellama',
        'mpt-', 'mpt_',
        'rwkv',
        'pythia',
        'cerebras',
        'neo',  # GPT-Neo
    ]

    # Full encoder-decoder transformers
    ENCODER_DECODER_PATTERNS = [
        't5', 't5-',
        'bart', 'mbart',
        'mt5',
        'pegasus',
        'prophetnet',
        'led',  # Longformer Encoder Decoder
        'bigbird_pegasus',
        'marian',
        'opus-mt',
        'nllb',  # No Language Left Behind
        'whisper',  # Speech (encoder-decoder)
        'speech',
        'wav2vec2_conformer',
    ]

    def __init__(self):
        # Compile patterns for efficient matching
        self._cnn_re = self._compile_patterns(self.CNN_PATTERNS)
        self._encoder_re = self._compile_patterns(self.ENCODER_PATTERNS)
        self._decoder_re = self._compile_patterns(self.DECODER_PATTERNS)
        self._enc_dec_re = self._compile_patterns(self.ENCODER_DECODER_PATTERNS)

    def _compile_patterns(self, patterns: list) -> re.Pattern:
        """Compile list of patterns into a single regex."""
        escaped = [re.escape(p) for p in patterns]
        return re.compile('|'.join(escaped), re.IGNORECASE)

    def classify(self, model_name: str) -> str:
        """
        Classify a model into its DNN architecture class.

        Args:
            model_name: Name of the model (e.g., 'resnet18', 'bert-base-uncased')

        Returns:
            One of: 'CNN', 'Encoder', 'Decoder', 'FullTransformer', 'Unknown'
        """
        name_lower = model_name.lower()

        # Check in order of specificity (encoder-decoder before encoder/decoder)
        if self._enc_dec_re.search(name_lower):
            return 'FullTransformer'

        if self._decoder_re.search(name_lower):
            return 'Decoder'

        if self._encoder_re.search(name_lower):
            return 'Encoder'

        if self._cnn_re.search(name_lower):
            return 'CNN'

        # Heuristics for unknown models
        if 'transformer' in name_lower:
            return 'Encoder'  # Default transformer to encoder

        if 'conv' in name_lower or 'cnn' in name_lower:
            return 'CNN'

        if 'attention' in name_lower:
            return 'Encoder'

        return 'Unknown'

    def classify_with_confidence(self, model_name: str) -> tuple:
        """
        Classify a model with confidence level.

        Returns:
            (class_name, confidence) where confidence is 'high', 'medium', or 'low'
        """
        name_lower = model_name.lower()

        # Direct pattern matches have high confidence
        if self._enc_dec_re.search(name_lower):
            return ('FullTransformer', 'high')

        if self._decoder_re.search(name_lower):
            return ('Decoder', 'high')

        if self._encoder_re.search(name_lower):
            return ('Encoder', 'high')

        if self._cnn_re.search(name_lower):
            return ('CNN', 'high')

        # Heuristic matches have medium confidence
        if 'transformer' in name_lower:
            return ('Encoder', 'medium')

        if 'conv' in name_lower or 'cnn' in name_lower:
            return ('CNN', 'medium')

        if 'attention' in name_lower:
            return ('Encoder', 'medium')

        return ('Unknown', 'low')

    def get_model_family(self, model_name: str) -> str:
        """
        Get the model family (e.g., 'ResNet', 'BERT', 'GPT-2').

        Args:
            model_name: Model name

        Returns:
            Family name with proper capitalization
        """
        name_lower = model_name.lower()

        # CNN families
        if 'resnet' in name_lower or 'resnext' in name_lower:
            return 'ResNet'
        if 'vgg' in name_lower:
            return 'VGG'
        if 'mobilenet' in name_lower:
            return 'MobileNet'
        if 'efficientnet' in name_lower:
            return 'EfficientNet'
        if 'densenet' in name_lower:
            return 'DenseNet'
        if 'shufflenet' in name_lower:
            return 'ShuffleNet'
        if 'squeezenet' in name_lower:
            return 'SqueezeNet'
        if 'regnet' in name_lower:
            return 'RegNet'
        if 'convnext' in name_lower:
            return 'ConvNeXt'
        if 'inception' in name_lower or 'googlenet' in name_lower:
            return 'Inception'
        if 'alexnet' in name_lower:
            return 'AlexNet'

        # Vision transformers
        if 'vit' in name_lower:
            return 'ViT'
        if 'swin' in name_lower:
            return 'Swin'
        if 'deit' in name_lower:
            return 'DeiT'
        if 'beit' in name_lower:
            return 'BEiT'

        # Language transformers
        if 'bert' in name_lower and 'distil' not in name_lower:
            if 'roberta' in name_lower:
                return 'RoBERTa'
            if 'albert' in name_lower:
                return 'ALBERT'
            if 'deberta' in name_lower:
                return 'DeBERTa'
            return 'BERT'
        if 'distilbert' in name_lower:
            return 'DistilBERT'
        if 'electra' in name_lower:
            return 'ELECTRA'

        if 'gpt' in name_lower:
            if 'neo' in name_lower:
                return 'GPT-Neo'
            if '2' in name_lower or '-2' in name_lower:
                return 'GPT-2'
            return 'GPT'
        if 'opt' in name_lower:
            return 'OPT'
        if 'llama' in name_lower:
            return 'LLaMA'
        if 'bloom' in name_lower:
            return 'BLOOM'
        if 'falcon' in name_lower:
            return 'Falcon'
        if 'mistral' in name_lower:
            return 'Mistral'

        if 't5' in name_lower:
            return 'T5'
        if 'bart' in name_lower:
            return 'BART'

        # Default to cleaned model name
        return model_name.split('_')[0].split('-')[0].capitalize()

    def is_vision_model(self, model_name: str) -> bool:
        """Check if model is a vision model (CNN or Vision Transformer)."""
        name_lower = model_name.lower()

        vision_patterns = [
            'resnet', 'vgg', 'mobilenet', 'efficientnet', 'densenet',
            'shufflenet', 'squeezenet', 'regnet', 'convnext', 'inception',
            'vit', 'swin', 'deit', 'beit', 'clip', 'pvt', 'twins',
            'alexnet', 'googlenet', 'yolo', 'unet', 'deeplabv3',
        ]

        return any(p in name_lower for p in vision_patterns)

    def is_language_model(self, model_name: str) -> bool:
        """Check if model is a language model."""
        name_lower = model_name.lower()

        language_patterns = [
            'bert', 'gpt', 'opt', 'llama', 'bloom', 'falcon', 'mistral',
            't5', 'bart', 'roberta', 'albert', 'electra', 'deberta',
            'xlm', 'xlnet', 'longformer', 'bigbird',
        ]

        return any(p in name_lower for p in language_patterns)

    @classmethod
    def list_patterns(cls) -> dict:
        """List all supported patterns by category."""
        return {
            'CNN': cls.CNN_PATTERNS,
            'Encoder': cls.ENCODER_PATTERNS,
            'Decoder': cls.DECODER_PATTERNS,
            'FullTransformer': cls.ENCODER_DECODER_PATTERNS,
        }
