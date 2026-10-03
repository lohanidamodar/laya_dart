// Download, sha256 verification and single-file extraction for the ONNX Runtime
// release archives. Shared by hook/build.dart and tool/pin_ort.dart.
import 'dart:io';

import 'package:archive/archive.dart';
import 'package:crypto/crypto.dart';

/// The sha256 of [file], as lowercase hex.
Future<String> sha256OfFile(File file) async =>
    (await sha256.bind(file.openRead()).first).toString();

/// Ensures [dest] holds the file at [url] with sha256 [expectedSha256].
///
/// An existing [dest] with the right digest is kept; anything else is
/// downloaded to `<dest>.part`, verified, and only then renamed into place, so
/// a partial or tampered download never sits at [dest].
Future<File> downloadVerified(Uri url, File dest, String expectedSha256) async {
  if (dest.existsSync() && await sha256OfFile(dest) == expectedSha256) {
    return dest;
  }
  dest.parent.createSync(recursive: true);
  final part = File('${dest.path}.part');
  final client = HttpClient();
  try {
    final response = await (await client.getUrl(url)).close();
    if (response.statusCode != 200) {
      throw HttpException('GET $url -> ${response.statusCode}', uri: url);
    }
    await response.pipe(part.openWrite());
  } finally {
    client.close();
  }
  final actual = await sha256OfFile(part);
  if (actual != expectedSha256) {
    part.deleteSync();
    throw StateError(
      'sha256 mismatch for $url: expected $expectedSha256, got $actual',
    );
  }
  if (dest.existsSync()) dest.deleteSync();
  part.renameSync(dest.path);
  return dest;
}

/// Extracts the regular file [entryPath] from the `.zip` or `.tgz` [archive]
/// to [dest], writing `<dest>.part` first and renaming it into place.
///
/// Throws [StateError] when the archive has no such regular file.
void extractEntry(File archive, String entryPath, File dest) {
  dest.parent.createSync(recursive: true);
  final part = File('${dest.path}.part');
  final temps = <String>[];
  final input = InputFileStream(archive.path);
  InputFileStream? tarInput;
  try {
    final Archive decoded;
    if (archive.path.endsWith('.zip')) {
      decoded = ZipDecoder().decodeStream(input);
    } else if (archive.path.endsWith('.tgz') ||
        archive.path.endsWith('.tar.gz')) {
      final tarPath = '${dest.path}.tar';
      temps.add(tarPath);
      final tarOut = OutputFileStream(tarPath);
      GZipDecoder().decodeStream(input, tarOut);
      tarOut.closeSync();
      tarInput = InputFileStream(tarPath);
      decoded = TarDecoder().decodeStream(tarInput);
    } else {
      throw StateError('unsupported archive type: ${archive.path}');
    }
    final entry = decoded.files.where(
      (f) => f.isFile && f.symbolicLink == null && f.name == entryPath,
    );
    if (entry.isEmpty) {
      throw StateError('${archive.path} has no regular file $entryPath');
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
    for (final t in temps) {
      final f = File(t);
      if (f.existsSync()) f.deleteSync();
    }
    if (part.existsSync()) part.deleteSync();
  }
}
