from .base import Decoder, Encoder
from .beta_tcvae import VAE
from .conv import ConvDecoder, ConvEncoder
from .identity import IdentityEncoder
from .normalize import NormalizeObservation
from .oracle import OracleEncoder
from .resnet import ResNet
from .table_env import TableEnvEncoder

IMAGE_ENCODERS = (ConvEncoder, ResNet, VAE)
