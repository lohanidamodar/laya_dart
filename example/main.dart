// Classifies one support ticket with the english checkpoint.
//
// Install the checkpoint first (about 1.6 GiB):
//
//   dart run laya_server:fetch english
//
// then run `dart run example/main.dart` from the directory holding `models/`.
import 'dart:convert';
import 'dart:io';

import 'package:laya_server/laya_server.dart';

Future<void> main() async {
  const modelDir = 'models';
  if (!ModelStore(modelDir).isInstalled('english')) {
    stdout.writeln(
      'The english checkpoint is not installed in $modelDir/. '
      'Fetch it with:\n\n  dart run laya_server:fetch english\n',
    );
    return;
  }

  final engine = LayaEngine(loader: LayaEngine.fileLoader(modelDir));
  try {
    final result = await engine.predict(
      'Customer: I was charged twice this month and I am furious. '
      'Fix it today or I cancel.',
      {
        'queue': {
          'type': 'choice',
          'instructions': 'Which team should handle this ticket?',
          'criteria': {
            'billing': 'payments, invoices, refunds',
            'tech': 'bugs and outages',
            'other': 'anything else',
          },
        },
        'urgency': {
          'type': 'score',
          'instructions': 'How upset is the customer?',
          'criteria': ['calm', 'firm', 'angry', 'furious'],
        },
      },
      const PredictOptions(),
    );
    stdout.writeln(const JsonEncoder.withIndent('  ').convert(result));
  } finally {
    await engine.close();
  }
}
