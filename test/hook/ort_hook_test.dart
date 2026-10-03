import 'dart:io';

import 'package:archive/archive.dart';
import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';

import '../../hook/ort_fetch.dart';
import '../../hook/ort_hook.dart';
import '../../hook/ort_release.dart';

/// The real ORT library the build hook bundled for this test run, so the
/// code-asset validator sees a genuine shared library.
File _realLibrary() {
  final name = OS.current.dylibFileName('onnxruntime');
  return Directory('.dart_tool')
      .listSync(recursive: true, followLinks: false)
      .whereType<File>()
      .firstWhere((f) => p.basename(f.path) == name);
}

Future<File> _noNetwork(Uri url, File dest, String sha) =>
    throw StateError('the hook tried to download $url');

/// Runs [buildOrt] through [testCodeBuildHook], returning the [HookError] it
/// threw (caught before `build()` would exit the process) and the output.
Future<(HookError?, BuildOutput)> _run({
  Map<String, Object?> defines = const {},
  OrtDownloader download = _noNetwork,
  OS? os,
  Architecture? arch,
  void Function(BuildOutput output)? inspect,
}) async {
  HookError? error;
  late BuildOutput result;
  final base = Directory.systemTemp.createTempSync('laya_hook');
  addTearDown(() => base.deleteSync(recursive: true));
  await testCodeBuildHook(
    targetOS: os,
    targetArchitecture: arch,
    userDefines: PackageUserDefines(
      workspacePubspec: PackageUserDefinesSource(
        defines: defines,
        basePath: base.uri,
      ),
    ),
    mainMethod: (args) => build(args, (input, output) async {
      try {
        await buildOrt(input, output, download: download);
      } on HookError catch (e) {
        error = e;
      }
    }),
    // The hook's output directory is deleted when testCodeBuildHook returns.
    check: (input, output) {
      result = output;
      inspect?.call(output);
    },
  );
  return (error, result);
}

Matcher _hinted(Type type) => isA<HookError>()
    .having((e) => e.runtimeType, 'type', type)
    .having(
      (e) => e.message,
      'message',
      allOf(
        contains('$ortLibDefine:'),
        contains('$bundleOrtDefine: false'),
        contains('LAYA_ORT_LIB'),
      ),
    );

void main() {
  test(
    'bundle_ort: false bundles nothing and does no network access',
    () async {
      final (error, output) = await _run(defines: {bundleOrtDefine: false});
      expect(error, isNull);
      expect(output.assets.code, isEmpty);
    },
  );

  test('ort_lib bundles the supplied library without downloading', () async {
    final lib = _realLibrary();
    final (error, output) = await _run(
      defines: {ortLibDefine: lib.absolute.path},
    );
    expect(error, isNull);
    final asset = output.assets.code.single;
    expect(asset.id, 'package:laya_server/onnxruntime');
    expect(asset.linkMode, isA<DynamicLoadingBundled>());
    expect(File.fromUri(asset.file!).absolute.path, lib.absolute.path);
  });

  test('ort_lib pointing nowhere is a BuildError with the opt-outs', () async {
    final (error, output) = await _run(defines: {ortLibDefine: 'no/such.dll'});
    expect(error, _hinted(BuildError));
    expect(error!.message, contains('such.dll'));
    expect(output.assets.code, isEmpty);
  });

  test('an unsupported target is a BuildError with the opt-outs', () async {
    final (error, _) = await _run(os: OS.linux, arch: Architecture.riscv64);
    expect(error, _hinted(BuildError));
    expect(error!.message, contains('linux-riscv64'));
  });

  test('a failed download is an InfraError with the opt-outs', () async {
    final (error, _) = await _run(
      download: (url, dest, sha) =>
          throw const SocketException('network is unreachable'),
    );
    expect(error, _hinted(InfraError));
    expect(error!.message, contains('network is unreachable'));
  });

  test('a digest mismatch is an InfraError with the opt-outs', () async {
    final (error, _) = await _run(
      download: (url, dest, sha) =>
          throw OrtFetchException('sha256 mismatch for $url'),
    );
    expect(error, _hinted(InfraError));
    expect(error!.message, contains('sha256 mismatch'));
  });

  group('with the host pin', () {
    final pin = ortRelease[(OS.current.name, Architecture.current.name)];

    /// A downloader that writes an archive holding [entries] instead.
    OrtDownloader fake(Map<String, List<int>> entries) =>
        (url, dest, sha) async {
          final archive = Archive();
          entries.forEach((k, v) => archive.add(ArchiveFile.bytes(k, v)));
          dest.parent.createSync(recursive: true);
          return dest..writeAsBytesSync(
            url.path.endsWith('.zip')
                ? ZipEncoder().encodeBytes(archive)
                : GZipEncoder().encodeBytes(TarEncoder().encodeBytes(archive)),
          );
        };

    test('extracts the pinned library and bundles it', () async {
      final bytes = _realLibrary().readAsBytesSync();
      late List<int> bundled;
      final (error, output) = await _run(
        download: fake({pin!.lib: bytes}),
        inspect: (o) => bundled = File.fromUri(
          o.assets.code.single.file!,
        ).readAsBytesSync(),
      );
      expect(error, isNull);
      expect(output.assets.code.single.id, 'package:laya_server/onnxruntime');
      expect(bundled, bytes);
    }, skip: pin == null ? 'no pin for this host' : false);

    test('an archive without the library is a BuildError', () async {
      final (error, output) = await _run(
        download: fake({
          'README': [1],
        }),
      );
      expect(error, _hinted(BuildError));
      expect(error!.message, contains(pin!.lib));
      expect(output.assets.code, isEmpty);
    }, skip: pin == null ? 'no pin for this host' : false);
  });
}
