from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Network(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VGG16: _ClassVar[Network]
    RESNET50: _ClassVar[Network]
    MOBILENETV2: _ClassVar[Network]
    VISIONTRANSFORMER: _ClassVar[Network]
VGG16: Network
RESNET50: Network
MOBILENETV2: Network
VISIONTRANSFORMER: Network

class SplitRequest(_message.Message):
    __slots__ = ("network", "partition_index", "tensor", "zero_point", "scale", "accelerator")
    NETWORK_FIELD_NUMBER: _ClassVar[int]
    PARTITION_INDEX_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    ZERO_POINT_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    ACCELERATOR_FIELD_NUMBER: _ClassVar[int]
    network: Network
    partition_index: int
    tensor: bytes
    zero_point: int
    scale: float
    accelerator: bool
    def __init__(self, network: _Optional[_Union[Network, str]] = ..., partition_index: _Optional[int] = ..., tensor: _Optional[bytes] = ..., zero_point: _Optional[int] = ..., scale: _Optional[float] = ..., accelerator: bool = ...) -> None: ...

class SplitResponse(_message.Message):
    __slots__ = ("label",)
    LABEL_FIELD_NUMBER: _ClassVar[int]
    label: str
    def __init__(self, label: _Optional[str] = ...) -> None: ...
