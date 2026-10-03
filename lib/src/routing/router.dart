/// Checkpoint routing, ported from laya's `Router._route` (laya 0.3.24).
library;

import '../model/checkpoint.dart';
import '../model/render.dart' show pyRepr;
import 'lang.dart';
import 'py_unicode.dart';

// Aliases people are likely to type (router.py:77).
const _aliases = {
  'en': 'english',
  'laya': 'english',
  'default': 'english',
  'multi': 'multilingual',
  'ml': 'multilingual',
  'laya-multilingual': 'multilingual',
  'typed': 'typed-decisions',
  'typed_decisions': 'typed-decisions',
  'laya-typed-decisions': 'typed-decisions',
  'decisions': 'typed-decisions',
};

// Question-id signatures of the typed-decisions workflows (router.py:86).
const _typedDecisionWorkflows = {
  'agent_trace_observability': {
    'action',
    'needs_review',
    'outcome',
    'risk',
    'urgency',
  },
  'customer_service': {
    'action',
    'category',
    'churn_risk',
    'needs_human',
    'urgency',
  },
  'invoice_processing': {
    'discrepancy_severity',
    'disposition',
    'duplicate',
    'matches_order',
    'urgency',
  },
  'security_incidents': {
    'credential_compromise',
    'disposition',
    'severity',
    'true_positive',
    'urgency',
  },
};

// Subtags that mean "the English checkpoint can read this" (router.py:162).
const _englishSubtags = {'en', 'eng', 'english'};

// Codes that name no language, so they answer nothing (router.py:171).
const _languageAgnosticCodes = {'c', 'posix', 'und', 'zxx', 'mul'};

String _pyLowerStr(String s) => String.fromCharCodes(pyLower(s.runes.toList()));

String _pyStripStr(String s) => String.fromCharCodes(pyStrip(s.runes.toList()));

/// Resolves a checkpoint name or alias, as laya's `normalise_name`: the name
/// is stripped and lowercased, then looked up among the aliases.
///
/// Throws an [ArgumentError] whose message is laya's `ValueError` message
/// when the name is neither a checkpoint nor an alias.
String normaliseCheckpointName(String name) {
  var key = _pyLowerStr(_pyStripStr(name));
  key = _aliases[key] ?? key;
  if (!CheckpointSpec.all.containsKey(key)) {
    throw ArgumentError(unknownCheckpointMessage(name));
  }
  return key;
}

/// laya `normalise_name`'s message for a [name] that is not a checkpoint or
/// alias. [name] may be any decoded JSON value: laya reads it through
/// `str()`, so no non-string names one.
String unknownCheckpointMessage(Object? name) {
  final names = CheckpointSpec.all.keys.toList()..sort();
  final aliases = _aliases.keys.toList()..sort();
  return 'unknown model ${pyRepr(name)}; choose one of ${pyRepr(names)} '
      '(or an alias: ${pyRepr(aliases)})';
}

/// The typed-decisions workflow whose question ids [questions] has exactly,
/// else null (laya `match_typed_decisions_workflow`).
String? _matchWorkflow(Map<String, Object?> questions) {
  final ids = questions.keys.toSet();
  for (final e in _typedDecisionWorkflows.entries) {
    if (ids.length == e.value.length && ids.containsAll(e.value)) return e.key;
  }
  return null;
}

/// True or false for a language code, or null when it identifies nothing
/// (laya `_english_from_code`).
bool? _englishFromCode(String? value) {
  if (value == null) return null;
  var code = _pyLowerStr(_pyStripStr(value));
  if (code.isEmpty) return null;
  code = code.split('.').first;
  final primary = code.replaceAll('_', '-').split('-').first;
  if (primary.isEmpty || _languageAgnosticCodes.contains(primary)) return null;
  return _englishSubtags.contains(primary);
}

/// Python `'%.0f' % x`: the nearest integer, an exact tie going to even.
String _percent0f(double x) {
  final floor = x.floorToDouble();
  final diff = x - floor; // exact for a finite double
  var n = floor.toInt();
  if (diff > 0.5 || (diff == 0.5 && n.isOdd)) n++;
  return '$n';
}

/// Which checkpoint answers a request, and why.
class RouteDecision {
  /// Creates a decision.
  const RouteDecision({
    required this.model,
    required this.repo,
    required this.reason,
    this.detection,
    this.workflow,
  });

  /// The checkpoint name: `english`, `multilingual` or `typed-decisions`.
  final String model;

  /// The checkpoint's repo as laya reports it ([CheckpointSpec.repo]).
  final String repo;

  /// laya's human-readable reason, verbatim.
  final String reason;

  /// The language analysis, when routing got as far as detection.
  final LanguageAnalysis? detection;

  /// The matched typed-decisions workflow, if any.
  final String? workflow;

  /// The response's `routing` block, in laya's key order.
  Map<String, Object?> toJson() => {
    'model': model,
    'repo': repo,
    'reason': reason,
    'detection': detection?.toJson(),
    'workflow': workflow,
  };
}

/// Routes a request to a checkpoint as laya's `Router._route` does.
///
/// Precedence: explicit `model` > explicit `task` > detected typed-decisions
/// workflow (only with [autoTaskDetection]) > explicit `lang` > `langGuess`
/// > detected script and language > [defaultModel].
class CheckpointRouter {
  /// Creates a router. [defaultModel] is normalised with
  /// [normaliseCheckpointName], so an unknown name throws an [ArgumentError].
  CheckpointRouter({
    String defaultModel = 'english',
    this.autoTaskDetection = false,
  }) : defaultModel = normaliseCheckpointName(defaultModel);

  /// The checkpoint for a state whose language nothing identifies.
  final String defaultModel;

  /// Whether a question schema matching a typed-decisions workflow exactly
  /// selects the typed-decisions checkpoint.
  final bool autoTaskDetection;

  /// Decides which checkpoint answers [state] and [questions].
  ///
  /// [state] is decoded JSON (a string, map or list) and [questions] the raw
  /// question map, whose ids are all that routing reads. [model] and [task]
  /// name a checkpoint and throw an [ArgumentError] when they name none;
  /// [lang] and [langGuess] are language codes such as `de` or `en_US.UTF-8`.
  RouteDecision route(
    Object? state,
    Map<String, Object?> questions, {
    String? model,
    String? task,
    String? lang,
    String? langGuess,
  }) {
    RouteDecision decision(
      String key,
      String reason, {
      LanguageAnalysis? detection,
      String? workflow,
    }) => RouteDecision(
      model: key,
      repo: CheckpointSpec.all[key]!.repo,
      reason: reason,
      detection: detection,
      workflow: workflow,
    );

    if (model != null) {
      return decision(
        normaliseCheckpointName(model),
        'explicit model=${pyRepr(model)}',
      );
    }
    if (task != null) {
      final typed = _pyLowerStr(task).replaceAll('-', '_') == 'typed_decisions';
      return decision(
        normaliseCheckpointName(typed ? 'typed-decisions' : task),
        'explicit task=${pyRepr(task)}',
      );
    }
    final workflow = _matchWorkflow(questions);
    if (workflow != null && autoTaskDetection) {
      return decision(
        'typed-decisions',
        'question ids match the ${pyRepr(workflow)} typed-decisions workflow',
        workflow: workflow,
      );
    }
    if (_englishFromCode(lang) case final english?) {
      return decision(
        english ? 'english' : 'multilingual',
        'explicit lang=${pyRepr(lang)}',
        workflow: workflow,
      );
    }
    if (_englishFromCode(langGuess) case final english?) {
      return decision(
        english ? 'english' : 'multilingual',
        'lang_guess: the caller identified this as '
        '${english ? 'English' : 'non-English'} text',
        workflow: workflow,
      );
    }

    final det = analyse(state);
    final String key;
    final String reason;
    if (det.script == 'unknown') {
      key = defaultModel;
      reason = 'no letters detected in state; using default ($key)';
    } else if (det.script != 'latin') {
      key = 'multilingual';
      reason =
          'non-Latin script (${det.script}, '
          '${_percent0f(100 * det.nonLatinFraction)}% of letters); '
          'the English checkpoint cannot read it';
    } else if (!det.isEnglish) {
      key = 'multilingual';
      final mixed = det.mixedSegment;
      if (mixed != null && mixed.isNotEmpty) {
        final head = String.fromCharCodes(mixed.runes.take(60));
        reason =
            'Latin script, mostly English, but a line or field reads as '
            '${pyRepr(det.language)} (${pyRepr(head)}); '
            'the English checkpoint cannot read it';
      } else if (det.language != null && det.language!.isNotEmpty) {
        reason =
            'Latin script but language looks like ${pyRepr(det.language)}, '
            'not English';
      } else {
        reason =
            'Latin script, language not identified but '
            '${_percent0f(100 * det.diacriticRate)}% non-English letters; '
            'not safe for the English checkpoint';
      }
    } else if (det.languageUndecided) {
      key = defaultModel;
      reason =
          'Latin script, language not identified and no non-English '
          'letters; using default ($key)';
    } else {
      key = 'english';
      reason = 'English Latin text';
    }
    return decision(key, reason, detection: det, workflow: workflow);
  }
}
