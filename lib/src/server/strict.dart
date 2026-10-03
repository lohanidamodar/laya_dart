/// The strict Jev projection of a response (`LAYA_JEV_STRICT`).
library;

/// Projects a full laya response onto the strict Jev `/v1/systemone`
/// contract, as laya `serve.py:128-163` `_project_jev_strict` does.
///
/// Keeps `model`, `answers` and the two `usage` token counts. Each answer
/// keeps only its type's fields: choice `type, choice, confidence,
/// probabilities`; score `type, score, confidence, probabilities, legend`;
/// noul `type, noul`. An answer of any other shape passes through unchanged.
/// Nothing is recomputed.
Map<String, Object?> projectJevStrict(Map<String, Object?> result) {
  final answers = <String, Object?>{};
  final given = result['answers'];
  if (given is Map) {
    for (final MapEntry(:key, :value) in given.entries) {
      final kind = value is Map ? value['type'] : null;
      answers['$key'] = switch (kind) {
        'choice' => {
          'type': 'choice',
          'choice': (value as Map)['choice'],
          'confidence': value['confidence'],
          'probabilities': value['probabilities'],
        },
        'score' => {
          'type': 'score',
          'score': (value as Map)['score'],
          'confidence': value['confidence'],
          'probabilities': value['probabilities'],
          'legend': value['legend'],
        },
        'noul' => {'type': 'noul', 'noul': (value as Map)['noul']},
        _ => value,
      };
    }
  }
  final usage = result['usage'] is Map
      ? result['usage']! as Map
      : const <Object?, Object?>{};
  return {
    'model': result['model'],
    'answers': answers,
    'usage': {
      'input_tokens': usage.containsKey('input_tokens')
          ? usage['input_tokens']
          : 0,
      'output_tokens': usage.containsKey('output_tokens')
          ? usage['output_tokens']
          : 0,
    },
  };
}
