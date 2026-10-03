// A minimal, hand-written dart:ffi binding to the ONNX Runtime C API.
//
// Written against include/onnxruntime_c_api.h of the v1.30.0 release
// (ORT_API_VERSION 30). Only the functions laya_server calls are typed; the
// OrtApi struct still declares every slot up to the last one used, in the
// header's order, because the struct is a table of function pointers and a
// field's offset is its position.
//
// ignore_for_file: non_constant_identifier_names

import 'dart:ffi';

import 'package:ffi/ffi.dart';

/// The `ORT_API_VERSION` of the header this binding was written against.
const ortApiVersion = 30;

/// `OrtErrorCode.ORT_OK`.
const ortOk = 0;

/// `OrtErrorCode.ORT_FAIL`.
const ortFail = 1;

/// `OrtErrorCode.ORT_INVALID_ARGUMENT`.
const ortInvalidArgument = 2;

/// `OrtErrorCode.ORT_NO_SUCHFILE`.
const ortNoSuchFile = 3;

/// `OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING`.
const ortLoggingLevelWarning = 2;

/// `GraphOptimizationLevel.ORT_ENABLE_ALL`.
const ortEnableAll = 99;

/// `OrtAllocatorType.OrtDeviceAllocator`.
const ortDeviceAllocator = 0;

/// `OrtAllocatorType.OrtArenaAllocator`.
const ortArenaAllocator = 1;

/// `OrtMemType.OrtMemTypeDefault`.
const ortMemTypeDefault = 0;

/// `ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT`.
const onnxTensorElementDataTypeFloat = 1;

/// `ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64`.
const onnxTensorElementDataTypeInt64 = 7;

/// `ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL`.
const onnxTensorElementDataTypeBool = 9;

/// Opaque `OrtStatus`.
final class OrtStatus extends Opaque {}

/// Opaque `OrtEnv`.
final class OrtEnv extends Opaque {}

/// Opaque `OrtSessionOptions`.
final class OrtSessionOptions extends Opaque {}

/// Opaque `OrtSession` (named apart from the Dart `OrtSession` wrapper).
final class OrtSessionHandle extends Opaque {}

/// Opaque `OrtRunOptions`.
final class OrtRunOptions extends Opaque {}

/// Opaque `OrtMemoryInfo`.
final class OrtMemoryInfo extends Opaque {}

/// Opaque `OrtValue`.
final class OrtValue extends Opaque {}

/// Opaque `OrtTensorTypeAndShapeInfo`.
final class OrtTensorTypeAndShapeInfo extends Opaque {}

/// Opaque `OrtAllocator`.
final class OrtAllocator extends Opaque {}

/// `OrtStatus* CreateStatus(OrtErrorCode code, const char* msg)`.
typedef CreateStatusNative =
    Pointer<OrtStatus> Function(Int32 code, Pointer<Utf8> msg);

/// `OrtErrorCode GetErrorCode(const OrtStatus* status)`.
typedef GetErrorCodeNative = Int32 Function(Pointer<OrtStatus> status);

/// `const char* GetErrorMessage(const OrtStatus* status)`.
typedef GetErrorMessageNative =
    Pointer<Utf8> Function(Pointer<OrtStatus> status);

/// `CreateEnv(OrtLoggingLevel, const char* logid, OrtEnv** out)`.
typedef CreateEnvNative =
    Pointer<OrtStatus> Function(
      Int32 level,
      Pointer<Utf8> logId,
      Pointer<Pointer<OrtEnv>> out,
    );

/// `CreateSession(const OrtEnv*, const ORTCHAR_T* model_path,
/// const OrtSessionOptions*, OrtSession** out)`.
///
/// `ORTCHAR_T` is `wchar_t` (UTF-16) on Windows and `char` elsewhere, so the
/// path is untyped here.
typedef CreateSessionNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtEnv> env,
      Pointer<Void> modelPath,
      Pointer<OrtSessionOptions> options,
      Pointer<Pointer<OrtSessionHandle>> out,
    );

/// `Run(OrtSession*, const OrtRunOptions*, const char* const* input_names,
/// const OrtValue* const* inputs, size_t input_len,
/// const char* const* output_names, size_t output_names_len,
/// OrtValue** outputs)`.
typedef RunNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtSessionHandle> session,
      Pointer<OrtRunOptions> runOptions,
      Pointer<Pointer<Utf8>> inputNames,
      Pointer<Pointer<OrtValue>> inputs,
      Size inputLen,
      Pointer<Pointer<Utf8>> outputNames,
      Size outputNamesLen,
      Pointer<Pointer<OrtValue>> outputs,
    );

/// `CreateSessionOptions(OrtSessionOptions** options)`.
typedef CreateSessionOptionsNative =
    Pointer<OrtStatus> Function(Pointer<Pointer<OrtSessionOptions>> out);

/// `SetSessionGraphOptimizationLevel(OrtSessionOptions*,
/// GraphOptimizationLevel)`.
typedef SetSessionGraphOptimizationLevelNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtSessionOptions> options,
      Int32 level,
    );

/// `SetIntraOpNumThreads(OrtSessionOptions*, int)`.
typedef SetIntraOpNumThreadsNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtSessionOptions> options,
      Int threads,
    );

/// `SessionGetInputCount` / `SessionGetOutputCount(const OrtSession*,
/// size_t* out)`.
typedef SessionGetCountNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtSessionHandle> session,
      Pointer<Size> out,
    );

/// `SessionGetInputName` / `SessionGetOutputName(const OrtSession*,
/// size_t index, OrtAllocator*, char** value)`.
typedef SessionGetNameNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtSessionHandle> session,
      Size index,
      Pointer<OrtAllocator> allocator,
      Pointer<Pointer<Utf8>> value,
    );

/// `CreateTensorWithDataAsOrtValue(const OrtMemoryInfo*, void* p_data,
/// size_t p_data_len, const int64_t* shape, size_t shape_len,
/// ONNXTensorElementDataType, OrtValue** out)`.
typedef CreateTensorWithDataAsOrtValueNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtMemoryInfo> info,
      Pointer<Void> data,
      Size dataLength,
      Pointer<Int64> shape,
      Size shapeLength,
      Int32 type,
      Pointer<Pointer<OrtValue>> out,
    );

/// `IsTensor(const OrtValue*, int* out)`.
typedef IsTensorNative =
    Pointer<OrtStatus> Function(Pointer<OrtValue> value, Pointer<Int> out);

/// `GetTensorMutableData(OrtValue*, void** out)`.
typedef GetTensorMutableDataNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtValue> value,
      Pointer<Pointer<Void>> out,
    );

/// `GetTensorElementType(const OrtTensorTypeAndShapeInfo*,
/// ONNXTensorElementDataType* out)`.
typedef GetTensorElementTypeNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtTensorTypeAndShapeInfo> info,
      Pointer<Int32> out,
    );

/// `GetDimensionsCount(const OrtTensorTypeAndShapeInfo*, size_t* out)`.
typedef GetDimensionsCountNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtTensorTypeAndShapeInfo> info,
      Pointer<Size> out,
    );

/// `GetDimensions(const OrtTensorTypeAndShapeInfo*, int64_t* dim_values,
/// size_t dim_values_length)`.
typedef GetDimensionsNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtTensorTypeAndShapeInfo> info,
      Pointer<Int64> dims,
      Size length,
    );

/// `GetTensorShapeElementCount(const OrtTensorTypeAndShapeInfo*,
/// size_t* out)`.
typedef GetTensorShapeElementCountNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtTensorTypeAndShapeInfo> info,
      Pointer<Size> out,
    );

/// `GetTensorTypeAndShape(const OrtValue*, OrtTensorTypeAndShapeInfo** out)`.
typedef GetTensorTypeAndShapeNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtValue> value,
      Pointer<Pointer<OrtTensorTypeAndShapeInfo>> out,
    );

/// `CreateMemoryInfo(const char* name, OrtAllocatorType, int id, OrtMemType,
/// OrtMemoryInfo** out)`.
typedef CreateMemoryInfoNative =
    Pointer<OrtStatus> Function(
      Pointer<Utf8> name,
      Int32 allocatorType,
      Int id,
      Int32 memType,
      Pointer<Pointer<OrtMemoryInfo>> out,
    );

/// `CreateCpuMemoryInfo(OrtAllocatorType, OrtMemType, OrtMemoryInfo** out)`.
typedef CreateCpuMemoryInfoNative =
    Pointer<OrtStatus> Function(
      Int32 allocatorType,
      Int32 memType,
      Pointer<Pointer<OrtMemoryInfo>> out,
    );

/// `MemoryInfoGetName(const OrtMemoryInfo*, const char** out)`.
typedef MemoryInfoGetNameNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtMemoryInfo> info,
      Pointer<Pointer<Utf8>> out,
    );

/// `MemoryInfoGetId(const OrtMemoryInfo*, int* out)`.
typedef MemoryInfoGetIdNative =
    Pointer<OrtStatus> Function(Pointer<OrtMemoryInfo> info, Pointer<Int> out);

/// `MemoryInfoGetMemType(const OrtMemoryInfo*, OrtMemType* out)`.
typedef MemoryInfoGetMemTypeNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtMemoryInfo> info,
      Pointer<Int32> out,
    );

/// `AllocatorFree(OrtAllocator*, void* p)`.
typedef AllocatorFreeNative =
    Pointer<OrtStatus> Function(
      Pointer<OrtAllocator> allocator,
      Pointer<Void> p,
    );

/// `GetAllocatorWithDefaultOptions(OrtAllocator** out)`.
typedef GetAllocatorWithDefaultOptionsNative =
    Pointer<OrtStatus> Function(Pointer<Pointer<OrtAllocator>> out);

/// `void Release<T>(T* input)`, from `ORT_CLASS_RELEASE`.
typedef ReleaseNative<T extends NativeType> = Void Function(Pointer<T> input);

/// `struct OrtApiBase`, which `OrtGetApiBase()` returns.
final class OrtApiBase extends Struct {
  /// `const OrtApi* GetApi(uint32_t version)`.
  external Pointer<NativeFunction<Pointer<OrtApi> Function(Uint32 version)>>
  GetApi;

  /// `const char* GetVersionString(void)`.
  external Pointer<NativeFunction<Pointer<Utf8> Function()>> GetVersionString;
}

/// `struct OrtApi` from the v1.30.0 `onnxruntime_c_api.h` (ORT_API_VERSION 30).
///
/// Every field from slot 0 up to the last slot used (100,
/// `ReleaseSessionOptions`) is declared in the header's order, since each
/// field's offset is its slot. Later slots are left undeclared; the struct is
/// only ever read through a pointer ORT owns.
final class OrtApi extends Struct {
  /// Slot 0: `CreateStatus`.
  external Pointer<NativeFunction<CreateStatusNative>> CreateStatus;

  /// Slot 1: `GetErrorCode`.
  external Pointer<NativeFunction<GetErrorCodeNative>> GetErrorCode;

  /// Slot 2: `GetErrorMessage`.
  external Pointer<NativeFunction<GetErrorMessageNative>> GetErrorMessage;

  /// Slot 3: `CreateEnv`.
  external Pointer<NativeFunction<CreateEnvNative>> CreateEnv;

  /// Slot 4: `CreateEnvWithCustomLogger` (unused; declared to keep the offsets).
  external Pointer<Void> CreateEnvWithCustomLogger;

  /// Slot 5: `EnableTelemetryEvents` (unused; declared to keep the offsets).
  external Pointer<Void> EnableTelemetryEvents;

  /// Slot 6: `DisableTelemetryEvents` (unused; declared to keep the offsets).
  external Pointer<Void> DisableTelemetryEvents;

  /// Slot 7: `CreateSession`.
  external Pointer<NativeFunction<CreateSessionNative>> CreateSession;

  /// Slot 8: `CreateSessionFromArray` (unused; declared to keep the offsets).
  external Pointer<Void> CreateSessionFromArray;

  /// Slot 9: `Run`.
  external Pointer<NativeFunction<RunNative>> Run;

  /// Slot 10: `CreateSessionOptions`.
  external Pointer<NativeFunction<CreateSessionOptionsNative>>
  CreateSessionOptions;

  /// Slot 11: `SetOptimizedModelFilePath` (unused; declared to keep the offsets).
  external Pointer<Void> SetOptimizedModelFilePath;

  /// Slot 12: `CloneSessionOptions` (unused; declared to keep the offsets).
  external Pointer<Void> CloneSessionOptions;

  /// Slot 13: `SetSessionExecutionMode` (unused; declared to keep the offsets).
  external Pointer<Void> SetSessionExecutionMode;

  /// Slot 14: `EnableProfiling` (unused; declared to keep the offsets).
  external Pointer<Void> EnableProfiling;

  /// Slot 15: `DisableProfiling` (unused; declared to keep the offsets).
  external Pointer<Void> DisableProfiling;

  /// Slot 16: `EnableMemPattern` (unused; declared to keep the offsets).
  external Pointer<Void> EnableMemPattern;

  /// Slot 17: `DisableMemPattern` (unused; declared to keep the offsets).
  external Pointer<Void> DisableMemPattern;

  /// Slot 18: `EnableCpuMemArena` (unused; declared to keep the offsets).
  external Pointer<Void> EnableCpuMemArena;

  /// Slot 19: `DisableCpuMemArena` (unused; declared to keep the offsets).
  external Pointer<Void> DisableCpuMemArena;

  /// Slot 20: `SetSessionLogId` (unused; declared to keep the offsets).
  external Pointer<Void> SetSessionLogId;

  /// Slot 21: `SetSessionLogVerbosityLevel` (unused; declared to keep the offsets).
  external Pointer<Void> SetSessionLogVerbosityLevel;

  /// Slot 22: `SetSessionLogSeverityLevel` (unused; declared to keep the offsets).
  external Pointer<Void> SetSessionLogSeverityLevel;

  /// Slot 23: `SetSessionGraphOptimizationLevel`.
  external Pointer<NativeFunction<SetSessionGraphOptimizationLevelNative>>
  SetSessionGraphOptimizationLevel;

  /// Slot 24: `SetIntraOpNumThreads`.
  external Pointer<NativeFunction<SetIntraOpNumThreadsNative>>
  SetIntraOpNumThreads;

  /// Slot 25: `SetInterOpNumThreads` (unused; declared to keep the offsets).
  external Pointer<Void> SetInterOpNumThreads;

  /// Slot 26: `CreateCustomOpDomain` (unused; declared to keep the offsets).
  external Pointer<Void> CreateCustomOpDomain;

  /// Slot 27: `CustomOpDomain_Add` (unused; declared to keep the offsets).
  external Pointer<Void> CustomOpDomain_Add;

  /// Slot 28: `AddCustomOpDomain` (unused; declared to keep the offsets).
  external Pointer<Void> AddCustomOpDomain;

  /// Slot 29: `RegisterCustomOpsLibrary` (unused; declared to keep the offsets).
  external Pointer<Void> RegisterCustomOpsLibrary;

  /// Slot 30: `SessionGetInputCount`.
  external Pointer<NativeFunction<SessionGetCountNative>> SessionGetInputCount;

  /// Slot 31: `SessionGetOutputCount`.
  external Pointer<NativeFunction<SessionGetCountNative>> SessionGetOutputCount;

  /// Slot 32: `SessionGetOverridableInitializerCount` (unused; declared to keep the offsets).
  external Pointer<Void> SessionGetOverridableInitializerCount;

  /// Slot 33: `SessionGetInputTypeInfo` (unused; declared to keep the offsets).
  external Pointer<Void> SessionGetInputTypeInfo;

  /// Slot 34: `SessionGetOutputTypeInfo` (unused; declared to keep the offsets).
  external Pointer<Void> SessionGetOutputTypeInfo;

  /// Slot 35: `SessionGetOverridableInitializerTypeInfo` (unused; declared to keep the offsets).
  external Pointer<Void> SessionGetOverridableInitializerTypeInfo;

  /// Slot 36: `SessionGetInputName`.
  external Pointer<NativeFunction<SessionGetNameNative>> SessionGetInputName;

  /// Slot 37: `SessionGetOutputName`.
  external Pointer<NativeFunction<SessionGetNameNative>> SessionGetOutputName;

  /// Slot 38: `SessionGetOverridableInitializerName` (unused; declared to keep the offsets).
  external Pointer<Void> SessionGetOverridableInitializerName;

  /// Slot 39: `CreateRunOptions` (unused; declared to keep the offsets).
  external Pointer<Void> CreateRunOptions;

  /// Slot 40: `RunOptionsSetRunLogVerbosityLevel` (unused; declared to keep the offsets).
  external Pointer<Void> RunOptionsSetRunLogVerbosityLevel;

  /// Slot 41: `RunOptionsSetRunLogSeverityLevel` (unused; declared to keep the offsets).
  external Pointer<Void> RunOptionsSetRunLogSeverityLevel;

  /// Slot 42: `RunOptionsSetRunTag` (unused; declared to keep the offsets).
  external Pointer<Void> RunOptionsSetRunTag;

  /// Slot 43: `RunOptionsGetRunLogVerbosityLevel` (unused; declared to keep the offsets).
  external Pointer<Void> RunOptionsGetRunLogVerbosityLevel;

  /// Slot 44: `RunOptionsGetRunLogSeverityLevel` (unused; declared to keep the offsets).
  external Pointer<Void> RunOptionsGetRunLogSeverityLevel;

  /// Slot 45: `RunOptionsGetRunTag` (unused; declared to keep the offsets).
  external Pointer<Void> RunOptionsGetRunTag;

  /// Slot 46: `RunOptionsSetTerminate` (unused; declared to keep the offsets).
  external Pointer<Void> RunOptionsSetTerminate;

  /// Slot 47: `RunOptionsUnsetTerminate` (unused; declared to keep the offsets).
  external Pointer<Void> RunOptionsUnsetTerminate;

  /// Slot 48: `CreateTensorAsOrtValue` (unused; declared to keep the offsets).
  external Pointer<Void> CreateTensorAsOrtValue;

  /// Slot 49: `CreateTensorWithDataAsOrtValue`.
  external Pointer<NativeFunction<CreateTensorWithDataAsOrtValueNative>>
  CreateTensorWithDataAsOrtValue;

  /// Slot 50: `IsTensor`.
  external Pointer<NativeFunction<IsTensorNative>> IsTensor;

  /// Slot 51: `GetTensorMutableData`.
  external Pointer<NativeFunction<GetTensorMutableDataNative>>
  GetTensorMutableData;

  /// Slot 52: `FillStringTensor` (unused; declared to keep the offsets).
  external Pointer<Void> FillStringTensor;

  /// Slot 53: `GetStringTensorDataLength` (unused; declared to keep the offsets).
  external Pointer<Void> GetStringTensorDataLength;

  /// Slot 54: `GetStringTensorContent` (unused; declared to keep the offsets).
  external Pointer<Void> GetStringTensorContent;

  /// Slot 55: `CastTypeInfoToTensorInfo` (unused; declared to keep the offsets).
  external Pointer<Void> CastTypeInfoToTensorInfo;

  /// Slot 56: `GetOnnxTypeFromTypeInfo` (unused; declared to keep the offsets).
  external Pointer<Void> GetOnnxTypeFromTypeInfo;

  /// Slot 57: `CreateTensorTypeAndShapeInfo` (unused; declared to keep the offsets).
  external Pointer<Void> CreateTensorTypeAndShapeInfo;

  /// Slot 58: `SetTensorElementType` (unused; declared to keep the offsets).
  external Pointer<Void> SetTensorElementType;

  /// Slot 59: `SetDimensions` (unused; declared to keep the offsets).
  external Pointer<Void> SetDimensions;

  /// Slot 60: `GetTensorElementType`.
  external Pointer<NativeFunction<GetTensorElementTypeNative>>
  GetTensorElementType;

  /// Slot 61: `GetDimensionsCount`.
  external Pointer<NativeFunction<GetDimensionsCountNative>> GetDimensionsCount;

  /// Slot 62: `GetDimensions`.
  external Pointer<NativeFunction<GetDimensionsNative>> GetDimensions;

  /// Slot 63: `GetSymbolicDimensions` (unused; declared to keep the offsets).
  external Pointer<Void> GetSymbolicDimensions;

  /// Slot 64: `GetTensorShapeElementCount`.
  external Pointer<NativeFunction<GetTensorShapeElementCountNative>>
  GetTensorShapeElementCount;

  /// Slot 65: `GetTensorTypeAndShape`.
  external Pointer<NativeFunction<GetTensorTypeAndShapeNative>>
  GetTensorTypeAndShape;

  /// Slot 66: `GetTypeInfo` (unused; declared to keep the offsets).
  external Pointer<Void> GetTypeInfo;

  /// Slot 67: `GetValueType` (unused; declared to keep the offsets).
  external Pointer<Void> GetValueType;

  /// Slot 68: `CreateMemoryInfo`.
  external Pointer<NativeFunction<CreateMemoryInfoNative>> CreateMemoryInfo;

  /// Slot 69: `CreateCpuMemoryInfo`.
  external Pointer<NativeFunction<CreateCpuMemoryInfoNative>>
  CreateCpuMemoryInfo;

  /// Slot 70: `CompareMemoryInfo` (unused; declared to keep the offsets).
  external Pointer<Void> CompareMemoryInfo;

  /// Slot 71: `MemoryInfoGetName`.
  external Pointer<NativeFunction<MemoryInfoGetNameNative>> MemoryInfoGetName;

  /// Slot 72: `MemoryInfoGetId`.
  external Pointer<NativeFunction<MemoryInfoGetIdNative>> MemoryInfoGetId;

  /// Slot 73: `MemoryInfoGetMemType`.
  external Pointer<NativeFunction<MemoryInfoGetMemTypeNative>>
  MemoryInfoGetMemType;

  /// Slot 74: `MemoryInfoGetType` (unused; declared to keep the offsets).
  external Pointer<Void> MemoryInfoGetType;

  /// Slot 75: `AllocatorAlloc` (unused; declared to keep the offsets).
  external Pointer<Void> AllocatorAlloc;

  /// Slot 76: `AllocatorFree`.
  external Pointer<NativeFunction<AllocatorFreeNative>> AllocatorFree;

  /// Slot 77: `AllocatorGetInfo` (unused; declared to keep the offsets).
  external Pointer<Void> AllocatorGetInfo;

  /// Slot 78: `GetAllocatorWithDefaultOptions`.
  external Pointer<NativeFunction<GetAllocatorWithDefaultOptionsNative>>
  GetAllocatorWithDefaultOptions;

  /// Slot 79: `AddFreeDimensionOverride` (unused; declared to keep the offsets).
  external Pointer<Void> AddFreeDimensionOverride;

  /// Slot 80: `GetValue` (unused; declared to keep the offsets).
  external Pointer<Void> GetValue;

  /// Slot 81: `GetValueCount` (unused; declared to keep the offsets).
  external Pointer<Void> GetValueCount;

  /// Slot 82: `CreateValue` (unused; declared to keep the offsets).
  external Pointer<Void> CreateValue;

  /// Slot 83: `CreateOpaqueValue` (unused; declared to keep the offsets).
  external Pointer<Void> CreateOpaqueValue;

  /// Slot 84: `GetOpaqueValue` (unused; declared to keep the offsets).
  external Pointer<Void> GetOpaqueValue;

  /// Slot 85: `KernelInfoGetAttribute_float` (unused; declared to keep the offsets).
  external Pointer<Void> KernelInfoGetAttribute_float;

  /// Slot 86: `KernelInfoGetAttribute_int64` (unused; declared to keep the offsets).
  external Pointer<Void> KernelInfoGetAttribute_int64;

  /// Slot 87: `KernelInfoGetAttribute_string` (unused; declared to keep the offsets).
  external Pointer<Void> KernelInfoGetAttribute_string;

  /// Slot 88: `KernelContext_GetInputCount` (unused; declared to keep the offsets).
  external Pointer<Void> KernelContext_GetInputCount;

  /// Slot 89: `KernelContext_GetOutputCount` (unused; declared to keep the offsets).
  external Pointer<Void> KernelContext_GetOutputCount;

  /// Slot 90: `KernelContext_GetInput` (unused; declared to keep the offsets).
  external Pointer<Void> KernelContext_GetInput;

  /// Slot 91: `KernelContext_GetOutput` (unused; declared to keep the offsets).
  external Pointer<Void> KernelContext_GetOutput;

  /// Slot 92: `ReleaseEnv`.
  external Pointer<NativeFunction<ReleaseNative<OrtEnv>>> ReleaseEnv;

  /// Slot 93: `ReleaseStatus`.
  external Pointer<NativeFunction<ReleaseNative<OrtStatus>>> ReleaseStatus;

  /// Slot 94: `ReleaseMemoryInfo`.
  external Pointer<NativeFunction<ReleaseNative<OrtMemoryInfo>>>
  ReleaseMemoryInfo;

  /// Slot 95: `ReleaseSession`.
  external Pointer<NativeFunction<ReleaseNative<OrtSessionHandle>>>
  ReleaseSession;

  /// Slot 96: `ReleaseValue`.
  external Pointer<NativeFunction<ReleaseNative<OrtValue>>> ReleaseValue;

  /// Slot 97: `ReleaseRunOptions` (unused; declared to keep the offsets).
  external Pointer<Void> ReleaseRunOptions;

  /// Slot 98: `ReleaseTypeInfo` (unused; declared to keep the offsets).
  external Pointer<Void> ReleaseTypeInfo;

  /// Slot 99: `ReleaseTensorTypeAndShapeInfo`.
  external Pointer<NativeFunction<ReleaseNative<OrtTensorTypeAndShapeInfo>>>
  ReleaseTensorTypeAndShapeInfo;

  /// Slot 100: `ReleaseSessionOptions`.
  external Pointer<NativeFunction<ReleaseNative<OrtSessionOptions>>>
  ReleaseSessionOptions;
}

/// The [OrtApi] functions laya_server calls, bound once to Dart closures.
///
/// Field names are the C names. Functions that return `OrtStatus*` return
/// `nullptr` on success; pass the result to the runtime's status check.
final class OrtFunctions {
  /// Binds the functions in [api].
  OrtFunctions(Pointer<OrtApi> api) : this._(api.ref);

  OrtFunctions._(OrtApi a)
    : CreateStatus = a.CreateStatus.asFunction(),
      GetErrorCode = a.GetErrorCode.asFunction(),
      GetErrorMessage = a.GetErrorMessage.asFunction(),
      CreateEnv = a.CreateEnv.asFunction(),
      CreateSession = a.CreateSession.asFunction(),
      Run = a.Run.asFunction(),
      CreateSessionOptions = a.CreateSessionOptions.asFunction(),
      SetSessionGraphOptimizationLevel =
          a.SetSessionGraphOptimizationLevel.asFunction(),
      SetIntraOpNumThreads = a.SetIntraOpNumThreads.asFunction(),
      SessionGetInputCount = a.SessionGetInputCount.asFunction(),
      SessionGetOutputCount = a.SessionGetOutputCount.asFunction(),
      SessionGetInputName = a.SessionGetInputName.asFunction(),
      SessionGetOutputName = a.SessionGetOutputName.asFunction(),
      CreateTensorWithDataAsOrtValue =
          a.CreateTensorWithDataAsOrtValue.asFunction(),
      IsTensor = a.IsTensor.asFunction(),
      GetTensorMutableData = a.GetTensorMutableData.asFunction(),
      GetTensorElementType = a.GetTensorElementType.asFunction(),
      GetDimensionsCount = a.GetDimensionsCount.asFunction(),
      GetDimensions = a.GetDimensions.asFunction(),
      GetTensorShapeElementCount = a.GetTensorShapeElementCount.asFunction(),
      GetTensorTypeAndShape = a.GetTensorTypeAndShape.asFunction(),
      CreateMemoryInfo = a.CreateMemoryInfo.asFunction(),
      CreateCpuMemoryInfo = a.CreateCpuMemoryInfo.asFunction(),
      MemoryInfoGetName = a.MemoryInfoGetName.asFunction(),
      MemoryInfoGetId = a.MemoryInfoGetId.asFunction(),
      MemoryInfoGetMemType = a.MemoryInfoGetMemType.asFunction(),
      AllocatorFree = a.AllocatorFree.asFunction(),
      GetAllocatorWithDefaultOptions =
          a.GetAllocatorWithDefaultOptions.asFunction(),
      ReleaseEnv = a.ReleaseEnv.asFunction(),
      ReleaseStatus = a.ReleaseStatus.asFunction(),
      ReleaseMemoryInfo = a.ReleaseMemoryInfo.asFunction(),
      ReleaseSession = a.ReleaseSession.asFunction(),
      ReleaseSessionPointer = a.ReleaseSession,
      ReleaseValue = a.ReleaseValue.asFunction(),
      ReleaseTensorTypeAndShapeInfo =
          a.ReleaseTensorTypeAndShapeInfo.asFunction(),
      ReleaseSessionOptions = a.ReleaseSessionOptions.asFunction();

  /// See [CreateStatusNative].
  final Pointer<OrtStatus> Function(int code, Pointer<Utf8> msg) CreateStatus;

  /// See [GetErrorCodeNative].
  final int Function(Pointer<OrtStatus> status) GetErrorCode;

  /// See [GetErrorMessageNative].
  final Pointer<Utf8> Function(Pointer<OrtStatus> status) GetErrorMessage;

  /// See [CreateEnvNative].
  final Pointer<OrtStatus> Function(
    int level,
    Pointer<Utf8> logId,
    Pointer<Pointer<OrtEnv>> out,
  )
  CreateEnv;

  /// See [CreateSessionNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtEnv> env,
    Pointer<Void> modelPath,
    Pointer<OrtSessionOptions> options,
    Pointer<Pointer<OrtSessionHandle>> out,
  )
  CreateSession;

  /// See [RunNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtSessionHandle> session,
    Pointer<OrtRunOptions> runOptions,
    Pointer<Pointer<Utf8>> inputNames,
    Pointer<Pointer<OrtValue>> inputs,
    int inputLen,
    Pointer<Pointer<Utf8>> outputNames,
    int outputNamesLen,
    Pointer<Pointer<OrtValue>> outputs,
  )
  Run;

  /// See [CreateSessionOptionsNative].
  final Pointer<OrtStatus> Function(Pointer<Pointer<OrtSessionOptions>> out)
  CreateSessionOptions;

  /// See [SetSessionGraphOptimizationLevelNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtSessionOptions> options,
    int level,
  )
  SetSessionGraphOptimizationLevel;

  /// See [SetIntraOpNumThreadsNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtSessionOptions> options,
    int threads,
  )
  SetIntraOpNumThreads;

  /// See [SessionGetCountNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtSessionHandle> session,
    Pointer<Size> out,
  )
  SessionGetInputCount;

  /// See [SessionGetCountNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtSessionHandle> session,
    Pointer<Size> out,
  )
  SessionGetOutputCount;

  /// See [SessionGetNameNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtSessionHandle> session,
    int index,
    Pointer<OrtAllocator> allocator,
    Pointer<Pointer<Utf8>> value,
  )
  SessionGetInputName;

  /// See [SessionGetNameNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtSessionHandle> session,
    int index,
    Pointer<OrtAllocator> allocator,
    Pointer<Pointer<Utf8>> value,
  )
  SessionGetOutputName;

  /// See [CreateTensorWithDataAsOrtValueNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtMemoryInfo> info,
    Pointer<Void> data,
    int dataLength,
    Pointer<Int64> shape,
    int shapeLength,
    int type,
    Pointer<Pointer<OrtValue>> out,
  )
  CreateTensorWithDataAsOrtValue;

  /// See [IsTensorNative].
  final Pointer<OrtStatus> Function(Pointer<OrtValue> value, Pointer<Int> out)
  IsTensor;

  /// See [GetTensorMutableDataNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtValue> value,
    Pointer<Pointer<Void>> out,
  )
  GetTensorMutableData;

  /// See [GetTensorElementTypeNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtTensorTypeAndShapeInfo> info,
    Pointer<Int32> out,
  )
  GetTensorElementType;

  /// See [GetDimensionsCountNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtTensorTypeAndShapeInfo> info,
    Pointer<Size> out,
  )
  GetDimensionsCount;

  /// See [GetDimensionsNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtTensorTypeAndShapeInfo> info,
    Pointer<Int64> dims,
    int length,
  )
  GetDimensions;

  /// See [GetTensorShapeElementCountNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtTensorTypeAndShapeInfo> info,
    Pointer<Size> out,
  )
  GetTensorShapeElementCount;

  /// See [GetTensorTypeAndShapeNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtValue> value,
    Pointer<Pointer<OrtTensorTypeAndShapeInfo>> out,
  )
  GetTensorTypeAndShape;

  /// See [CreateMemoryInfoNative].
  final Pointer<OrtStatus> Function(
    Pointer<Utf8> name,
    int allocatorType,
    int id,
    int memType,
    Pointer<Pointer<OrtMemoryInfo>> out,
  )
  CreateMemoryInfo;

  /// See [CreateCpuMemoryInfoNative].
  final Pointer<OrtStatus> Function(
    int allocatorType,
    int memType,
    Pointer<Pointer<OrtMemoryInfo>> out,
  )
  CreateCpuMemoryInfo;

  /// See [MemoryInfoGetNameNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtMemoryInfo> info,
    Pointer<Pointer<Utf8>> out,
  )
  MemoryInfoGetName;

  /// See [MemoryInfoGetIdNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtMemoryInfo> info,
    Pointer<Int> out,
  )
  MemoryInfoGetId;

  /// See [MemoryInfoGetMemTypeNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtMemoryInfo> info,
    Pointer<Int32> out,
  )
  MemoryInfoGetMemType;

  /// See [AllocatorFreeNative].
  final Pointer<OrtStatus> Function(
    Pointer<OrtAllocator> allocator,
    Pointer<Void> p,
  )
  AllocatorFree;

  /// See [GetAllocatorWithDefaultOptionsNative].
  final Pointer<OrtStatus> Function(Pointer<Pointer<OrtAllocator>> out)
  GetAllocatorWithDefaultOptions;

  /// `ReleaseEnv`.
  final void Function(Pointer<OrtEnv> input) ReleaseEnv;

  /// `ReleaseStatus`.
  final void Function(Pointer<OrtStatus> input) ReleaseStatus;

  /// `ReleaseMemoryInfo`.
  final void Function(Pointer<OrtMemoryInfo> input) ReleaseMemoryInfo;

  /// `ReleaseSession`.
  final void Function(Pointer<OrtSessionHandle> input) ReleaseSession;

  /// The raw `ReleaseSession` pointer, for a `NativeFinalizer`.
  final Pointer<NativeFunction<ReleaseNative<OrtSessionHandle>>>
  ReleaseSessionPointer;

  /// `ReleaseValue`.
  final void Function(Pointer<OrtValue> input) ReleaseValue;

  /// `ReleaseTensorTypeAndShapeInfo`.
  final void Function(Pointer<OrtTensorTypeAndShapeInfo> input)
  ReleaseTensorTypeAndShapeInfo;

  /// `ReleaseSessionOptions`.
  final void Function(Pointer<OrtSessionOptions> input) ReleaseSessionOptions;
}
