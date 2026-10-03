import 'dart:convert';

import 'package:laya_server/src/server/strict.dart';
import 'package:test/test.dart';

void main() {
  // The full-payload sample from laya docs/http-api.md (research.md 1.2), plus
  // a noul answer.
  final full = <String, Object?>{
    'model': 'laya-rl-agent',
    'answers': {
      'queue': {
        'type': 'choice',
        'choice': 'billing',
        'probabilities': {'billing': 0.9519, 'tech': 0.0327, 'other': 0.0154},
        'confidence': 0.797,
        'answer_confidence': 0.9519,
        'action': {'act_probability': 1.0},
      },
      'urgency': {
        'type': 'score',
        'score': 1.6994,
        'legend': {'0': 'calm', '1': 'firm', '2': 'angry', '3': 'furious'},
        'probabilities': {'0': 0.0249, '1': 0.4136, '2': 0.3985, '3': 0.1629},
        'confidence': 0.1925,
        'answer_confidence': 0.4136,
        'action': {'act_probability': 1.0},
      },
      'refund': {
        'type': 'noul',
        'noul': 0.7357,
        'confidence': 0.7357,
        'answer_confidence': 0.7357,
        'action': {'act_probability': 0.1943},
        'low_confidence': true,
        'abstention': 'abstained',
        'abstention_threshold': 0.8,
      },
    },
    'usage': {
      'input_tokens': 83,
      'output_tokens': 0,
      'state_tokens': 12,
      'state_tokens_dropped': 0,
      'truncated': false,
      'truncated_questions': <Object?>[],
    },
    'routing': {'model': 'english', 'repo': 'convaiinnovations/laya'},
  };

  test('projects onto the strict Jev contract, in laya key order', () {
    expect(
      jsonEncode(projectJevStrict(full)),
      jsonEncode({
        'model': 'laya-rl-agent',
        'answers': {
          'queue': {
            'type': 'choice',
            'choice': 'billing',
            'confidence': 0.797,
            'probabilities': {
              'billing': 0.9519,
              'tech': 0.0327,
              'other': 0.0154,
            },
          },
          'urgency': {
            'type': 'score',
            'score': 1.6994,
            'confidence': 0.1925,
            'probabilities': {
              '0': 0.0249,
              '1': 0.4136,
              '2': 0.3985,
              '3': 0.1629,
            },
            'legend': {'0': 'calm', '1': 'firm', '2': 'angry', '3': 'furious'},
          },
          'refund': {'type': 'noul', 'noul': 0.7357},
        },
        'usage': {'input_tokens': 83, 'output_tokens': 0},
      }),
    );
  });

  test('an answer of unknown shape passes through unchanged', () {
    final odd = {
      'type': 'ranking',
      'order': <Object?>['a', 'b'],
    };
    final out = projectJevStrict({
      'model': 'm',
      'answers': {'x': odd, 'y': 'not a map'},
      'usage': {'input_tokens': 3},
    });
    expect(out['answers'], {'x': odd, 'y': 'not a map'});
    expect(out['usage'], {'input_tokens': 3, 'output_tokens': 0});
  });

  test('missing answers and usage become empty', () {
    expect(projectJevStrict({'model': 'm'}), {
      'model': 'm',
      'answers': <String, Object?>{},
      'usage': {'input_tokens': 0, 'output_tokens': 0},
    });
  });
}
