/// HTTP-mappable failures.
library;

/// An HTTP-mappable failure: [status] and the client-safe [detail].
class ApiException implements Exception {
  /// Creates a failure with an HTTP [status], a client-safe [detail] and
  /// optional response [headers].
  const ApiException(this.status, this.detail, {this.headers = const {}});

  /// The HTTP status code.
  final int status;

  /// The message returned to the client.
  final String detail;

  /// Extra response headers.
  final Map<String, String> headers;

  @override
  String toString() => 'ApiException($status): $detail';
}
