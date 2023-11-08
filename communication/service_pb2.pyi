from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Network(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    VGG16: _ClassVar[Network]
    RESNET50: _ClassVar[Network]
    MOBILENETV2: _ClassVar[Network]
VGG16: Network
RESNET50: Network
MOBILENETV2: Network

class SplitRequest(_message.Message):
    __slots__ = ["network", "partition_index", "tensor", "quantized_tail"]
    NETWORK_FIELD_NUMBER: _ClassVar[int]
    PARTITION_INDEX_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    QUANTIZED_TAIL_FIELD_NUMBER: _ClassVar[int]
    network: Network
    partition_index: int
    tensor: bytes
    quantized_tail: bool
    def __init__(self, network: _Optional[_Union[Network, str]] = ..., partition_index: _Optional[int] = ..., tensor: _Optional[bytes] = ..., quantized_tail: bool = ...) -> None: ...

class SplitResponse(_message.Message):
    __slots__ = ["classes", "server_time"]
    CLASSES_FIELD_NUMBER: _ClassVar[int]
    SERVER_TIME_FIELD_NUMBER: _ClassVar[int]
    classes: _containers.RepeatedScalarFieldContainer[str]
    server_time: int
    def __init__(self, classes: _Optional[_Iterable[str]] = ..., server_time: _Optional[int] = ...) -> None: ...
