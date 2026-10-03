// Download, sha256 verification and single-file extraction for the ONNX Runtime
// release archives. Shared by hook/build.dart and tool/pin_ort.dart.
import 'dart:async';
import 'dart:io';

import 'package:archive/archive.dart';
import 'package:crypto/crypto.dart';

/// A download or extraction failure, with a one-line [message].
class OrtFetchException implements Exception {
  /// A failure described by [message].
  OrtFetchException(this.message);

  /// What went wrong.
  final String message;

  @override
  String toString() => 'OrtFetchException: $message';
}

/// How long to wait for a connection, and for each chunk of a download.
const fetchTimeout = Duration(seconds: 60);

/// The sha256 of [file], as lowercase hex.
Future<String> sha256OfFile(File file) async =>
    (await sha256.bind(file.openRead()).first).toString();

/// Ensures [dest] holds the file at [url] with sha256 [expectedSha256].
///
/// An existing [dest] with the right digest is kept without any network
/// access; anything else is downloaded to `<dest>.part`, verified, and only
/// then renamed into place, so a partial or tampered download never sits at
/// [dest].
///
/// Throws [OrtFetchException] on an HTTP error or a digest mismatch, and
/// [IOException] or [TimeoutException] when the transfer fails or stalls for
/// [timeout].
Future<File> downloadVerified(
  Uri url,
  File dest,
  String expectedSha256, {
  Duration timeout = fetchTimeout,
}) async {
  if (dest.existsSync() && await sha256OfFile(dest) == expectedSha256) {
    return dest;
  }
  dest.parent.createSync(recursive: true);
  final part = File('${dest.path}.part');
  final client = HttpClient()..connectionTimeout = timeout;
  try {
    final response = await (await client.getUrl(url)).close().timeout(timeout);
    if (response.statusCode != 200) {
      throw OrtFetchException('GET $url returned HTTP ${response.statusCode}');
    }
    await response.timeout(timeout).pipe(part.openWrite());
  } on Object {
    if (part.existsSync()) part.deleteSync();
    rethrow;
  } finally {
    client.close(force: true);
  }
  final actual = await sha256OfFile(part);
  if (actual != expectedSha256) {
    part.deleteSync();
    throw OrtFetchException(
      'sha256 mismatch for $url: expected $expectedSha256, got $actual',
    );
  }
  if (dest.existsSync()) dest.deleteSync();
  part.renameSync(dest.path);
  return dest;
}

/// The archive entry name without a leading `./`, which some tar writers add.
String _normalise(String name) {
  var n = name;
  while (n.startsWith('./')) {
    n = n.substring(2);
  }
  return n;
}

/// Extracts the regular file [entryPath] from the `.zip` or `.tgz` [archive]
/// to [dest], writing `<dest>.part` first and renaming it into place.
///
/// Entry names are compared without a leading `./`. Throws
/// [OrtFetchException] when the archive is unreadable or has no such regular
/// file.
void extractEntry(File archive, String entryPath, File dest) {
  dest.parent.createSync(recursive: true);
  final wanted = _normalise(entryPath);
  final part = File('${dest.path}.part');
  final tar = File('${dest.path}.tar');
  final input = InputFileStream(archive.path);
  InputFileStream? tarInput;
  try {
    final Archive decoded;
    try {
      if (archive.path.endsWith('.zip')) {
        decoded = ZipDecoder().decodeStream(input);
      } else if (archive.path.endsWith('.tgz') ||
          archive.path.endsWith('.tar.gz')) {
        final tarOut = OutputFileStream(tar.path);
        try {
          GZipDecoder().decodeStream(input, tarOut);
        } finally {
          tarOut.closeSync();
        }
        tarInput = InputFileStream(tar.path);
        decoded = TarDecoder().decodeStream(tarInput);
      } else {
        throw OrtFetchException('unsupported archive type: ${archive.path}');
      }
    } on OrtFetchException {
      rethrow;
    } on Exception catch (e) {
      throw OrtFetchException('could not read ${archive.path}: $e');
    }
    final entry = decoded.files.where(
      (f) => f.isFile && f.symbolicLink == null && _normalise(f.name) == wanted,
    );
    if (entry.isEmpty) {
      throw OrtFetchException('${archive.path} has no regular file $wanted');
    }
    final out = OutputFileStream(part.path);
    try {
      entry.first.writeContent(out);
    } finally {
      out.closeSync();
    }
    if (dest.existsSync()) dest.deleteSync();
    part.renameSync(dest.path);
  } finally {
    input.closeSync();
    tarInput?.closeSync();
    if (tar.existsSync()) tar.deleteSync();
    if (part.existsSync()) part.deleteSync();
  }
}
