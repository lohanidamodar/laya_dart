import 'dart:convert';
import 'dart:io';

import 'package:laya_server/src/model/checkpoint.dart';
import 'package:laya_server/src/model/question.dart';
import 'package:laya_server/src/model/sequence.dart';
import 'package:laya_server/src/tokenizer/tokenizer.dart';
import 'package:test/test.dart';

import '../support/checkpoints.dart';

void main() {
  final fixture =
      jsonDecode(File('test/fixtures/sequence.json').readAsStringSync()) as Map;
  final cases = (fixture['cases'] as List).cast<Map<String, Object?>>();
  final checkpoints = fixture['checkpoints'] as Map;
  final cf =
      jsonDecode(File('tool/reference/cases.json').readAsStringSync()) as Map;
  final requests = (cf['requests'] as List).cast<Map<String, Object?>>();
  final states = cf['states'] as List;

  for (final name in ['english', 'multilingual', 'typed-decisions']) {
    test('$name spec matches the fixture', () {
      final spec = CheckpointSpec.all[name]!;
      final f = checkpoints[name] as Map;
      expect(spec.maxLen, f['max_len']);
      expect(spec.headMaxLen, f['head_max_len']);
    });

    test('$name rows match laya build_sequence', () async {
      final tok = await LayaTokenizer.load(tokenizerPath(name));
      final spec = CheckpointSpec.all[name]!;
      final f = checkpoints[name] as Map;
      expect(tok.clsId, f['cls_token_id']);
      expect(tok.sepId, f['sep_token_id']);
      expect(tok.maskId, f['mask_token_id']);
      expect(tok.maskToken, f['mask_token']);
      var n = 0;
      for (final c in cases.where((c) => c['checkpoint'] == name)) {
        if (c.containsKey('error')) continue;
        final req = requests[c['request_index']! as int];
        final Object state = states[req['state']! as int] as Object;
        final qid = c['qid']! as String;
        final q = Question.parse(qid, (req['questions']! as Map)[qid]);
        final why = '$name request ${c['request_index']} $qid';
        expect(state is List, c['truncate_left'], reason: why);
        final row = buildSequence(
          tok,
          state,
          q,
          maxLen: spec.maxLen,
          headMaxLen: spec.headMaxLen,
          truncateLeft: state is List,
        );
        expect(row.ids, c['ids'], reason: why);
        expect(row.markers, c['markers'], reason: why);
        expect(row.options.toJson(), c['stats'], reason: why);
        expect(row.truncation.toJson(), c['truncation'], reason: why);
        expect(
          row.options.toJson().keys.toList(),
          (c['stats']! as Map).keys.toList(),
        );
        // state_room must agree with the room build_sequence left.
        final room = stateRoom(
          tok,
          q,
          maxLen: spec.maxLen,
          headMaxLen: spec.headMaxLen,
        );
        final t = row.truncation;
        expect(t.stateTokensUsed, lessThanOrEqualTo(room), reason: why);
        // Reusing pre-encoded state ids gives the same row.
        final again = buildSequence(
          tok,
          state,
          q,
          maxLen: spec.maxLen,
          headMaxLen: spec.headMaxLen,
          truncateLeft: state is List,
          stateIds: encodeState(tok, state),
        );
        expect(again.ids, row.ids, reason: why);
        n++;
      }
      expect(n, greaterThan(0));
    }, skip: tokenizerSkip(name));
  }
}
