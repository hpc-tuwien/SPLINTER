# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: service.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rservice.proto\"\x98\x01\n\x0cSplitRequest\x12\x19\n\x07network\x18\x01 \x01(\x0e\x32\x08.Network\x12\x17\n\x0fpartition_index\x18\x02 \x01(\r\x12\x0e\n\x06tensor\x18\x03 \x01(\x0c\x12\x17\n\nzero_point\x18\x04 \x01(\x11H\x00\x88\x01\x01\x12\x12\n\x05scale\x18\x05 \x01(\x02H\x01\x88\x01\x01\x42\r\n\x0b_zero_pointB\x08\n\x06_scale\"5\n\rSplitResponse\x12\x0f\n\x07\x63lasses\x18\x01 \x03(\t\x12\x13\n\x0bserver_time\x18\x02 \x01(\x04*3\n\x07Network\x12\t\n\x05VGG16\x10\x00\x12\x0c\n\x08RESNET50\x10\x01\x12\x0f\n\x0bMOBILENETV2\x10\x02\x32=\n\x0cSplitService\x12-\n\x0cSplitCompute\x12\r.SplitRequest\x1a\x0e.SplitResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'service_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_NETWORK']._serialized_start=227
  _globals['_NETWORK']._serialized_end=278
  _globals['_SPLITREQUEST']._serialized_start=18
  _globals['_SPLITREQUEST']._serialized_end=170
  _globals['_SPLITRESPONSE']._serialized_start=172
  _globals['_SPLITRESPONSE']._serialized_end=225
  _globals['_SPLITSERVICE']._serialized_start=280
  _globals['_SPLITSERVICE']._serialized_end=341
# @@protoc_insertion_point(module_scope)
