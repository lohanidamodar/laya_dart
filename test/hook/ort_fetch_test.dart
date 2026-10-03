import 'dart:convert';
import 'dart:io';

import 'package:archive/archive.dart';
import 'package:crypto/crypto.dart';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';

import '../../hook/ort_fetch.dart';

/// Writes a .tgz holding `lib/libonnxruntime.dylib` (plus a decoy and a
/// symlink) under [prefix], e.g. './' as the osx-arm64 release does.
File _tgz(Directory dir, String prefix) {
  final tar = Archive()
    ..add(ArchiveFile.bytes('${prefix}ort/README', utf8.encode('readme')))
    ..add(
      ArchiveFile.bytes(
        '${prefix}ort/lib/libonnxruntime.1.30.0.dylib',
        utf8.encode('the library'),
      ),
    );
  final bytes = GZipEncoder().encodeBytes(TarEncoder().encodeBytes(tar));
  return File(p.join(dir.path, 'ort.tgz'))..writeAsBytesSync(bytes);
}

void main() {
  late Directory dir;
  setUp(() => dir = Directory.systemTemp.createTempSync('laya_fetch'));
  tearDown(() => dir.deleteSync(recursive: true));

  group('extractEntry', () {
    for (final prefix in ['', './']) {
      test("finds the entry when tar names start with '$prefix'", () {
        final out = File(p.join(dir.path, 'out', 'libonnxruntime.dylib'));
        extractEntry(
          _tgz(dir, prefix),
          'ort/lib/libonnxruntime.1.30.0.dylib',
          out,
        );
        expect(out.readAsStringSync(), 'the library');
        expect(File('${out.path}.part').existsSync(), isFalse);
        expect(File('${out.path}.tar').existsSync(), isFalse);
      });
    }

    test('a missing entry is an OrtFetchException naming it', () {
      final out = File(p.join(dir.path, 'out', 'lib'));
      expect(
        () => extractEntry(_tgz(dir, './'), 'ort/lib/nope.dll', out),
        throwsA(
          isA<OrtFetchException>().having(
            (e) => e.message,
            'message',
            contains('ort/lib/nope.dll'),
          ),
        ),
      );
      expect(out.existsSync(), isFalse);
    });

    test('a corrupt archive is an OrtFetchException', () {
      final bad = File(p.join(dir.path, 'bad.tgz'))
        ..writeAsBytesSync([1, 2, 3, 4]);
      expect(
        () => extractEntry(bad, 'x', File(p.join(dir.path, 'x'))),
        throwsA(isA<OrtFetchException>()),
      );
    });
  });

  group('downloadVerified', () {
    late HttpServer server;
    late Uri url;
    final body = utf8.encode('archive bytes');
    final digest = sha256.convert(body).toString();

    setUp(() async {
      server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      server.listen((r) {
        r.response
          ..add(body)
          ..close();
      });
      url = Uri.parse('http://127.0.0.1:${server.port}/ort.zip');
    });
    tearDown(() => server.close(force: true));

    test('a wrong digest is an OrtFetchException and leaves nothing', () async {
      final dest = File(p.join(dir.path, 'ort.zip'));
      await expectLater(
        downloadVerified(url, dest, '0' * 64),
        throwsA(
          isA<OrtFetchException>().having(
            (e) => e.message,
            'message',
            allOf(contains('sha256 mismatch'), contains(digest)),
          ),
        ),
      );
      expect(dest.existsSync(), isFalse);
      expect(File('${dest.path}.part').existsSync(), isFalse);
    });

    test('the right digest lands the file', () async {
      final dest = File(p.join(dir.path, 'ort.zip'));
      await downloadVerified(url, dest, digest);
      expect(dest.readAsBytesSync(), body);
    });

    test('an existing verified file needs no network', () async {
      final dest = File(p.join(dir.path, 'ort.zip'))..writeAsBytesSync(body);
      await server.close(force: true);
      await downloadVerified(url, dest, digest);
      expect(dest.readAsBytesSync(), body);
    });

    test('an HTTP error is an OrtFetchException', () async {
      await server.close(force: true);
      server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      server.listen((r) {
        r.response
          ..statusCode = 404
          ..close();
      });
      final missing = Uri.parse('http://127.0.0.1:${server.port}/x.zip');
      await expectLater(
        downloadVerified(missing, File(p.join(dir.path, 'x.zip')), digest),
        throwsA(
          isA<OrtFetchException>().having(
            (e) => e.message,
            'message',
            contains('404'),
          ),
        ),
      );
    });
  });
}
