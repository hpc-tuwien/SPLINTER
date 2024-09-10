# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: service.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import duration_pb2 as google_dot_protobuf_dot_duration__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rservice.proto\x1a\x1egoogle/protobuf/duration.proto\"\xb9\x01\n\x12SessionInitRequest\x12\x14\n\x0cnum_requests\x18\x01 \x01(\r\x12\x19\n\x07network\x18\x02 \x01(\x0e\x32\x08.Network\x12\x17\n\x0fpartition_index\x18\x03 \x01(\r\x12\x17\n\nzero_point\x18\x04 \x01(\x11H\x00\x88\x01\x01\x12\x12\n\x05scale\x18\x05 \x01(\x02H\x01\x88\x01\x01\x12\x13\n\x0b\x61\x63\x63\x65lerator\x18\x06 \x01(\x08\x42\r\n\x0b_zero_pointB\x08\n\x06_scale\".\n\x13SessionInitResponse\x12\x17\n\x06status\x18\x01 \x01(\x0e\x32\x07.Status\"*\n\x0cSplitRequest\x12\n\n\x02id\x18\x01 \x01(\r\x12\x0e\n\x06tensor\x18\x02 \x01(\x0c\"*\n\rSplitResponse\x12\n\n\x02id\x18\x01 \x01(\r\x12\r\n\x05label\x18\x02 \x01(\t\"\x10\n\x0eMetricsRequest\"\x97\x01\n\x07Metrics\x12\x31\n\x0eserver_latency\x18\x01 \x01(\x0b\x32\x19.google.protobuf.Duration\x12\x17\n\x0f\x63pu_utilization\x18\x02 \x01(\x02\x12\x17\n\x0fgpu_utilization\x18\x03 \x01(\x02\x12\x12\n\ngpu_energy\x18\x04 \x01(\x02\x12\x13\n\x0bnode_energy\x18\x05 \x01(\x02*i\n\x07Network\x12\x17\n\x13NETWORK_UNSPECIFIED\x10\x00\x12\n\n\x06VGG_16\x10\x01\x12\x0e\n\nRES_NET_50\x10\x02\x12\x11\n\rMOBILE_NET_V2\x10\x03\x12\x16\n\x12VISION_TRANSFORMER\x10\x04*:\n\x06Status\x12\x16\n\x12STATUS_UNSPECIFIED\x10\x00\x12\t\n\x05READY\x10\x01\x12\r\n\tNOT_READY\x10\x02\x32\xaa\x01\n\x0cSplitService\x12>\n\x11InitializeSession\x12\x13.SessionInitRequest\x1a\x14.SessionInitResponse\x12\x31\n\x0cSplitCompute\x12\r.SplitRequest\x1a\x0e.SplitResponse(\x01\x30\x01\x12\'\n\nGetMetrics\x12\x0f.MetricsRequest\x1a\x08.Metricsb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_NETWORK']._serialized_start=545
  _globals['_NETWORK']._serialized_end=650
  _globals['_STATUS']._serialized_start=652
  _globals['_STATUS']._serialized_end=710
  _globals['_SESSIONINITREQUEST']._serialized_start=50
  _globals['_SESSIONINITREQUEST']._serialized_end=235
  _globals['_SESSIONINITRESPONSE']._serialized_start=237
  _globals['_SESSIONINITRESPONSE']._serialized_end=283
  _globals['_SPLITREQUEST']._serialized_start=285
  _globals['_SPLITREQUEST']._serialized_end=327
  _globals['_SPLITRESPONSE']._serialized_start=329
  _globals['_SPLITRESPONSE']._serialized_end=371
  _globals['_METRICSREQUEST']._serialized_start=373
  _globals['_METRICSREQUEST']._serialized_end=389
  _globals['_METRICS']._serialized_start=392
  _globals['_METRICS']._serialized_end=543
  _globals['_SPLITSERVICE']._serialized_start=713
  _globals['_SPLITSERVICE']._serialized_end=883
# @@protoc_insertion_point(module_scope)
