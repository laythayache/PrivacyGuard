# Commercial Use and Deployment Notes

PrivacyGuard is distributed under the [MIT License](LICENSE), which permits
commercial use subject to the license terms. This repository does not claim to
operate a SaaS product, model marketplace, hardware appliance, compliance-audit
service, paid support program, or the former `privacyguard.dev` contact channel.

## Before offering a deployment

A commercial deployment should define and validate at least:

- the problem, processing purpose, users, jurisdictions, and system owner;
- the model artifact, label mapping, supported targets, thresholds, and failure
  behavior;
- representative evaluation footage and class-specific false negatives;
- hardware, runtime, input, masking, storage, and performance configuration;
- access control, encryption, retention, deletion, monitoring, and incident
  response in the surrounding system;
- human review and the response when detections are uncertain or missed;
- contracts, notices, consent or other lawful basis, and qualified legal review
  where required; and
- maintenance, model updates, regression testing, handover, and support terms.

Local processing and visual masking can be useful technical controls, but they
do not by themselves establish anonymity or compliance. Do not sell or describe
the software as automatically satisfying GDPR, CCPA, HIPAA, or another legal
framework.

## Reproducible evidence

Do not price, market, or scope a deployment around the historical 25–30 FPS
observation. The retained repository does not contain the complete configuration
needed to reproduce that result. Benchmark the intended model and full pipeline
on the target hardware using [BENCHMARKS.md](BENCHMARKS.md) and
[PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md).

Any public case study should separate:

- library capabilities from the surrounding application;
- a controlled test from production operation;
- a technical control from a legal conclusion; and
- demonstrated adoption from a proposed business model.
