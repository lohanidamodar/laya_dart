// An ONNX Runtime inference session over the hand-written C API binding.
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:meta/meta.dart';

import 'ort_api.dart';
import 'ort_library.dart';

/// A named input tensor for [OrtSession.run].
sealed class OrtInput {
  /// The tensor's dimensions; their product must equal the element count.
  List<int> get shape;

  int get _length;
}

/// An int64 tensor input.
final class OrtInt64Input extends OrtInput {
  /// An int64 tensor of [shape] holding [data] in row-major order.
  OrtInt64Input(this.data, this.shape);

  /// The elements, row-major.
  final Int64List data;

  @override
  final List<int> shape;

  @override
  int get _length => data.length;
}

/// A bool tensor input, one byte per element: zero is false, anything else
/// true.
final class OrtBoolInput extends OrtInput {
  /// A bool tensor of [shape] holding [data] in row-major order.
  OrtBoolInput(this.data, this.shape);

  /// The elements, row-major.
  final Uint8List data;

  @override
  final List<int> shape;

  @override
  int get _length => data.length;
}

/// A float32 output tensor, copied out of ORT's memory.
class OrtTensorF32 {
  /// A tensor of [shape] holding [data] in row-major order.
  OrtTensorF32(this.data, this.shape);

  /// The elements, row-major.
  final Float32List data;

  /// The tensor's dimensions.
  final List<int> shape;
}

/// A loaded graph. Not thread-safe across isolates; [close] releases native
/// memory.
///
/// A session that is never closed is released when it is garbage collected,
/// but that is a backstop: close sessions explicitly.
final class OrtSession implements Finalizable {
  OrtSession._(this._runtime, this._handle, this.inputNames, this.outputNames) {
    _runtime.sessionFinalizer.attach(this, _handle.cast(), detach: this);
  }

  /// Opens the graph at [modelPath]; use [OrtRuntime.openSession].
  @internal
  factory OrtSession.open(
    OrtRuntime runtime,
    String modelPath, {
    int? intraOpThreads,
  }) {
    final f = runtime.functions;
    return using((arena) {
      final optionsOut = arena<Pointer<OrtSessionOptions>>();
      runtime.check(f.CreateSessionOptions(optionsOut));
      final options = optionsOut.value;
      try {
        runtime.check(
          f.SetSessionGraphOptimizationLevel(options, ortEnableAll),
        );
        if (intraOpThreads != null) {
          runtime.check(f.SetIntraOpNumThreads(options, intraOpThreads));
        }
        // ORTCHAR_T is wchar_t (UTF-16) on Windows and char (UTF-8) elsewhere.
        final Pointer<Void> path = Platform.isWindows
            ? modelPath.toNativeUtf16(allocator: arena).cast()
            : modelPath.toNativeUtf8(allocator: arena).cast();
        final sessionOut = arena<Pointer<OrtSessionHandle>>();
        runtime.check(f.CreateSession(runtime.env, path, options, sessionOut));
        final handle = sessionOut.value;
        try {
          return OrtSession._(
            runtime,
            handle,
            _names(runtime, handle, arena, input: true),
            _names(runtime, handle, arena, input: false),
          );
        } catch (_) {
          f.ReleaseSession(handle);
          rethrow;
        }
      } finally {
        f.ReleaseSessionOptions(options);
      }
    });
  }

  final OrtRuntime _runtime;
  final Pointer<OrtSessionHandle> _handle;
  bool _closed = false;

  /// The graph's input names, in graph order.
  final List<String> inputNames;

  /// The graph's output names, in graph order.
  final List<String> outputNames;

  static List<String> _names(
    OrtRuntime runtime,
    Pointer<OrtSessionHandle> handle,
    Arena arena, {
    required bool input,
  }) {
    final f = runtime.functions;
    final allocatorOut = arena<Pointer<OrtAllocator>>();
    // The default allocator is owned by ORT and never released.
    runtime.check(f.GetAllocatorWithDefaultOptions(allocatorOut));
    final allocator = allocatorOut.value;
    final count = arena<Size>();
    runtime.check(
      (input ? f.SessionGetInputCount : f.SessionGetOutputCount)(handle, count),
    );
    final getName = input ? f.SessionGetInputName : f.SessionGetOutputName;
    final nameOut = arena<Pointer<Utf8>>();
    return [
      for (var i = 0; i < count.value; i++)
        () {
          runtime.check(getName(handle, i, allocator, nameOut));
          try {
            return nameOut.value.toDartString();
          } finally {
            runtime.check(f.AllocatorFree(allocator, nameOut.value.cast()));
          }
        }(),
    ];
  }

  /// Runs with named inputs; returns float32 outputs by name with their
  /// shapes.
  ///
  /// Throws [ArgumentError] when an input's shape does not match its data,
  /// [StateError] after [close], and [OrtException] for anything ORT rejects
  /// (an unknown input name, a missing input, a wrong element type) and for a
  /// non-float32 output.
  Map<String, OrtTensorF32> run(Map<String, OrtInput> inputs) {
    if (_closed) throw StateError('OrtSession.run after close');
    for (final MapEntry(:key, :value) in inputs.entries) {
      if (value.shape.any((d) => d < 0)) {
        throw ArgumentError.value(value.shape, key, 'negative dimension');
      }
      final count = value.shape.fold(1, (a, b) => a * b);
      if (count != value._length) {
        throw ArgumentError.value(
          value.shape,
          key,
          'shape holds $count elements but the data has ${value._length}',
        );
      }
    }

    final f = _runtime.functions;
    final arena = Arena();
    final inputValues = arena<Pointer<OrtValue>>(_atLeastOne(inputs.length));
    final outputValues = arena<Pointer<OrtValue>>(
      _atLeastOne(outputNames.length),
    );
    Pointer<OrtMemoryInfo> memory = nullptr;
    try {
      final memoryOut = arena<Pointer<OrtMemoryInfo>>();
      _runtime.check(
        f.CreateCpuMemoryInfo(ortDeviceAllocator, ortMemTypeDefault, memoryOut),
      );
      memory = memoryOut.value;

      final inputNamesC = arena<Pointer<Utf8>>(_atLeastOne(inputs.length));
      var i = 0;
      for (final MapEntry(:key, :value) in inputs.entries) {
        inputNamesC[i] = key.toNativeUtf8(allocator: arena);
        final n = value._length;
        final Pointer<Void> data;
        final int bytes;
        final int type;
        switch (value) {
          case OrtInt64Input(data: final src):
            final buffer = arena<Int64>(_atLeastOne(n));
            buffer.asTypedList(n).setAll(0, src);
            (data, bytes, type) = (
              buffer.cast(),
              n * 8,
              onnxTensorElementDataTypeInt64,
            );
          case OrtBoolInput(data: final src):
            final buffer = arena<Uint8>(_atLeastOne(n));
            final view = buffer.asTypedList(n);
            for (var j = 0; j < n; j++) {
              view[j] = src[j] == 0 ? 0 : 1;
            }
            (data, bytes, type) = (
              buffer.cast(),
              n,
              onnxTensorElementDataTypeBool,
            );
        }
        final rank = value.shape.length;
        final shape = arena<Int64>(_atLeastOne(rank));
        shape.asTypedList(rank).setAll(0, value.shape);
        // The value borrows `data`, which the arena frees after the value is
        // released in `finally`.
        _runtime.check(
          f.CreateTensorWithDataAsOrtValue(
            memory,
            data,
            bytes,
            shape,
            rank,
            type,
            inputValues + i,
          ),
        );
        i++;
      }

      final outputNamesC = arena<Pointer<Utf8>>(
        _atLeastOne(outputNames.length),
      );
      for (var j = 0; j < outputNames.length; j++) {
        outputNamesC[j] = outputNames[j].toNativeUtf8(allocator: arena);
      }
      _runtime.check(
        f.Run(
          _handle,
          nullptr,
          inputNamesC,
          inputValues,
          inputs.length,
          outputNamesC,
          outputNames.length,
          outputValues,
        ),
      );
      return {
        for (var j = 0; j < outputNames.length; j++)
          outputNames[j]: _readF32(outputNames[j], outputValues[j], arena),
      };
    } finally {
      for (var j = 0; j < outputNames.length; j++) {
        if (outputValues[j] != nullptr) f.ReleaseValue(outputValues[j]);
      }
      for (var j = 0; j < inputs.length; j++) {
        if (inputValues[j] != nullptr) f.ReleaseValue(inputValues[j]);
      }
      if (memory != nullptr) f.ReleaseMemoryInfo(memory);
      arena.releaseAll();
    }
  }

  OrtTensorF32 _readF32(String name, Pointer<OrtValue> value, Arena arena) {
    final f = _runtime.functions;
    final infoOut = arena<Pointer<OrtTensorTypeAndShapeInfo>>();
    _runtime.check(f.GetTensorTypeAndShape(value, infoOut));
    final info = infoOut.value;
    try {
      final type = arena<Int32>();
      _runtime.check(f.GetTensorElementType(info, type));
      if (type.value != onnxTensorElementDataTypeFloat) {
        throw OrtException(
          ortFail,
          'output $name has ONNX element type ${type.value}; '
          'only float32 (1) outputs are supported',
        );
      }
      final rankOut = arena<Size>();
      _runtime.check(f.GetDimensionsCount(info, rankOut));
      final rank = rankOut.value;
      final dims = arena<Int64>(_atLeastOne(rank));
      _runtime.check(f.GetDimensions(info, dims, rank));
      final shape = List<int>.unmodifiable(dims.asTypedList(rank));
      final count = shape.fold(1, (a, b) => a * b);
      final dataOut = arena<Pointer<Void>>();
      _runtime.check(f.GetTensorMutableData(value, dataOut));
      final data = count == 0
          ? Float32List(0)
          : Float32List.fromList(
              dataOut.value.cast<Float>().asTypedList(count),
            );
      return OrtTensorF32(data, shape);
    } finally {
      f.ReleaseTensorTypeAndShapeInfo(info);
    }
  }

  /// Releases the native session. Idempotent; [run] throws afterwards.
  void close() {
    if (_closed) return;
    _closed = true;
    _runtime.sessionFinalizer.detach(this);
    _runtime.functions.ReleaseSession(_handle);
  }
}

int _atLeastOne(int n) => n < 1 ? 1 : n;
