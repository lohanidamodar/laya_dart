/// Downloads and verifies the pinned checkpoint files.
library;

import 'dart:async';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;

import 'pins.g.dart';

/// A pinned file of a checkpoint on the Hugging Face hub.
class PinnedFile {
  /// Creates a pin: [path] inside [repo] at [revision], [size] bytes long with
  /// the lowercase hex [sha256].
  const PinnedFile(this.repo, this.revision, this.path, this.size, this.sha256);

  /// The hub repository, for example `onnx-community/laya-ONNX`.
  final String repo;

  /// The pinned commit of [repo].
  final String revision;

  /// The file's path in the repository; it is also its path under
  /// `<root>/<checkpoint>/`.
  final String path;

  /// The size in bytes.
  final int size;

  /// The lowercase hex sha256 of the whole file.
  final String sha256;

  @override
  String toString() => '$repo@$revision/$path';
}

/// Installs and verifies checkpoint files under `root/<checkpoint>/`.
///
/// The layout mirrors the hub repository, which is what
/// `LayaEngine.fileLoader` reads: `tokenizer.json`, `config.json` and
/// `onnx/model.onnx` plus `onnx/model.onnx_data` (the `model_fp16` pair for
/// the `fp16` variant).
class ModelStore {
  /// A store rooted at [root].
  ///
  /// [client] performs the downloads (a default client is created and closed
  /// per [install] when omitted). [hubBase] replaces `https://huggingface.co`,
  /// for tests and mirrors. [token] is the Hugging Face token, defaulting to
  /// the `HF_TOKEN` environment variable; it is sent as a bearer token to the
  /// [hubBase] host only, never to a redirect target on another host. [pins]
  /// replaces the built-in pinned table, for tests.
  ModelStore(
    this.root, {
    http.Client? client,
    Uri? hubBase,
    String? token,
    Map<String, List<PinnedFile>>? pins,
  }) : _client = client,
       hubBase = hubBase ?? Uri.parse('https://huggingface.co'),
       _token = _nonEmpty(token) ?? _nonEmpty(Platform.environment['HF_TOKEN']),
       _pins = pins ?? pinnedFiles;

  /// The directory holding one folder per checkpoint.
  final String root;

  /// The hub base URL.
  final Uri hubBase;

  final http.Client? _client;
  final String? _token;
  final Map<String, List<PinnedFile>> _pins;

  static String? _nonEmpty(String? s) => s == null || s.isEmpty ? null : s;

  /// Files a checkpoint needs for [variant] (`fp32` or `fp16`).
  ///
  /// Throws [ArgumentError] for an unknown checkpoint or variant.
  List<PinnedFile> filesFor(String checkpoint, {String variant = 'fp32'}) {
    final all =
        _pins[checkpoint] ??
        (throw ArgumentError.value(
          checkpoint,
          'checkpoint',
          'is not a checkpoint',
        ));
    final prefix = switch (variant) {
      'fp32' => 'onnx/model.onnx',
      'fp16' => 'onnx/model_fp16.onnx',
      _ => throw ArgumentError.value(
        variant,
        'variant',
        'must be fp32 or fp16',
      ),
    };
    return [
      for (final f in all)
        if (!f.path.startsWith('onnx/') || f.path.startsWith(prefix)) f,
    ];
  }

  /// The local path of [file] of [checkpoint].
  String pathOf(String checkpoint, PinnedFile file) =>
      p.joinAll([root, checkpoint, ...file.path.split('/')]);

  /// True when every file is present with the pinned size (the fast check used
  /// at startup; [verify] hashes).
  bool isInstalled(String checkpoint, {String variant = 'fp32'}) {
    for (final f in filesFor(checkpoint, variant: variant)) {
      final file = File(pathOf(checkpoint, f));
      if (!file.existsSync() || file.lengthSync() != f.size) return false;
    }
    return true;
  }

  /// Hashes every installed file and returns the local paths that are missing
  /// or whose sha256 differs from the pin.
  Future<List<String>> verify(
    String checkpoint, {
    String variant = 'fp32',
  }) async {
    final bad = <String>[];
    for (final f in filesFor(checkpoint, variant: variant)) {
      final path = pathOf(checkpoint, f);
      final file = File(path);
      if (!file.existsSync() || await _sha256Of(file) != f.sha256) {
        bad.add(path);
      }
    }
    return bad;
  }

  /// Downloads the files [checkpoint] needs for [variant] and verifies each
  /// one's sha256.
  ///
  /// A file already present with its pinned size is skipped (run [verify] to
  /// hash it). Each download goes to `<path>.part`, resuming a partial one
  /// with a `Range` request, and is renamed into place only once its sha256
  /// matches. Throws [StateError] naming the path on a mismatch, after
  /// deleting the bad `.part`; [HttpException] on a failed request.
  /// [onProgress] receives the file's relative path and its bytes done of
  /// total.
  Future<void> install(
    String checkpoint, {
    String variant = 'fp32',
    void Function(String path, int done, int total)? onProgress,
  }) async {
    final files = filesFor(checkpoint, variant: variant);
    final client = _client ?? http.Client();
    try {
      for (final f in files) {
        final path = pathOf(checkpoint, f);
        final existing = File(path);
        if (existing.existsSync() && existing.lengthSync() == f.size) {
          onProgress?.call(f.path, f.size, f.size);
          continue;
        }
        await _download(client, f, path, onProgress);
      }
    } finally {
      if (_client == null) client.close();
    }
  }

  Future<void> _download(
    http.Client client,
    PinnedFile f,
    String path,
    void Function(String path, int done, int total)? onProgress,
  ) async {
    final part = File('$path.part');
    part.parent.createSync(recursive: true);
    var have = part.existsSync() ? part.lengthSync() : 0;
    if (have > f.size) {
      part.deleteSync();
      have = 0;
    }
    if (have < f.size) {
      have = await _fetchInto(client, f, part, have, onProgress);
    }
    if (have < f.size) {
      throw StateError(
        '$path: download ended at $have of ${f.size} bytes; '
        'run it again to resume',
      );
    }
    if (await _sha256Of(part) != f.sha256) {
      part.deleteSync();
      throw StateError(
        '$path: sha256 does not match the pin for ${f.path} of ${f.repo}',
      );
    }
    final target = File(path);
    if (target.existsSync()) target.deleteSync();
    part.renameSync(path);
  }

  /// Streams [f] into [part] from byte [have] on; returns the bytes held.
  Future<int> _fetchInto(
    http.Client client,
    PinnedFile f,
    File part,
    int have,
    void Function(String path, int done, int total)? onProgress,
  ) async {
    final response = await _get(client, _resolveUrl(f), {
      if (have > 0) 'Range': 'bytes=$have-',
    });
    final status = response.statusCode;
    if (status == 416 && have > 0) {
      // The partial file is unusable: start over.
      await response.stream.drain<void>();
      part.deleteSync();
      return _fetchInto(client, f, part, 0, onProgress);
    }
    if (status != 200 && status != 206) {
      await response.stream.drain<void>();
      throw HttpException('GET ${f.path} of ${f.repo}: HTTP $status');
    }
    if (status == 206 && have > 0) {
      final range = response.headers['content-range'];
      if (range != null && !range.startsWith('bytes $have-')) {
        await response.stream.drain<void>();
        throw HttpException('GET ${f.path}: unexpected Content-Range $range');
      }
    } else {
      have = 0; // a 200 means the server ignored the Range: from zero.
    }
    final sink = part.openWrite(
      mode: have > 0 ? FileMode.append : FileMode.write,
    );
    try {
      await for (final chunk in response.stream) {
        sink.add(chunk);
        have += chunk.length;
        onProgress?.call(f.path, have, f.size);
      }
    } finally {
      await sink.close();
    }
    return have;
  }

  Uri _resolveUrl(PinnedFile f) => hubBase.replace(
    pathSegments: [
      ...hubBase.pathSegments.where((s) => s.isNotEmpty),
      ...f.repo.split('/'),
      'resolve',
      f.revision,
      ...f.path.split('/'),
    ],
  );

  /// GET [url] following redirects by hand, so that `Range` survives every hop
  /// and the token reaches only the [hubBase] host.
  Future<http.StreamedResponse> _get(
    http.Client client,
    Uri url,
    Map<String, String> headers,
  ) async {
    for (var hop = 0; hop < 8; hop++) {
      final request = http.Request('GET', url)
        ..followRedirects = false
        ..headers.addAll(headers);
      if (_token != null && _sameOrigin(url, hubBase)) {
        request.headers['Authorization'] = 'Bearer $_token';
      }
      final response = await client.send(request);
      final location = response.headers['location'];
      if (response.statusCode >= 300 &&
          response.statusCode < 400 &&
          location != null) {
        await response.stream.drain<void>();
        url = url.resolve(location);
        continue;
      }
      return response;
    }
    throw HttpException('too many redirects', uri: url);
  }

  static bool _sameOrigin(Uri a, Uri b) =>
      a.scheme == b.scheme && a.host == b.host && a.port == b.port;

  static Future<String> _sha256Of(File file) async =>
      (await sha256.bind(file.openRead()).first).toString();
}
