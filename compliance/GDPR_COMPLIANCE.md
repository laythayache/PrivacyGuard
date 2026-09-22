# GDPR Deployment Considerations for PrivacyGuard

> **Engineering guidance, not legal advice or certification.** This checklist does not determine whether the GDPR applies or whether a deployment complies with it. The controller or processor remains responsible for the complete processing operation and should obtain qualified legal or data-protection advice where appropriate.

Last reviewed: 22 September 2026.

## What PrivacyGuard can contribute

PrivacyGuard can be configured to run a compatible ONNX detector locally and apply blur, pixelation, or solid masking to selected detected regions. This may support data-minimisation, privacy-by-design, and security objectives by reducing the visual information released downstream.

That contribution is limited:

- raw frames still enter the process and exist in device memory;
- missed, occluded, incorrectly labelled, or out-of-distribution regions will not be masked correctly;
- video context, clothing, gait, audio, timestamps, location, metadata, or other indirect identifiers may still identify a person;
- blur and pixelation are not encryption, and no masking method proves that output is anonymous;
- model choice, class mapping, thresholds, input quality, and masking configuration materially affect the result;
- optional operation logs document software events, not legal compliance.

The [ICO’s anonymisation guidance](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/data-sharing/anonymisation/how-do-we-ensure-anonymisation-is-effective/) explains that identifiability must be assessed in context. Treat processed output as personal data unless a documented assessment supports a different conclusion.

## What the library does not provide

PrivacyGuard does not select a lawful basis or purpose, issue privacy notices, collect or manage consent, answer data-subject requests, set retention periods, encrypt storage, control user access, manage processor contracts, govern international transfers, perform breach notification, or complete a data-protection impact assessment.

It also does not guarantee that a deployment has no network calls. Evaluate the complete application, operating system, model delivery process, monitoring stack, storage, and infrastructure—not only this package.

## Deployment checklist

### 1. Scope and accountability

- [ ] Identify the controller, processors, sub-processors, system owner, and data-protection contact.
- [ ] Document the processing purpose, affected people, data categories, sources, recipients, locations, and lifecycle.
- [ ] Determine the applicable jurisdictions and lawful basis with qualified advice.
- [ ] Maintain the required processing records and contracts for the complete system.
- [ ] Assign an owner and review date for this assessment.

### 2. Necessity, proportionality, and transparency

- [ ] Confirm that video processing is necessary for the stated purpose and that a less intrusive approach is insufficient.
- [ ] Limit cameras, capture areas, resolution, frame rate, classes, outputs, and retention to what the purpose requires.
- [ ] Provide accurate notices describing raw capture, local processing, masking limits, storage, recipients, and rights.
- [ ] Do not describe output as anonymous without a documented identifiability assessment.
- [ ] Do not describe PrivacyGuard as making a system GDPR compliant.

### 3. Risk assessment and data protection by design

- [ ] Determine whether a data-protection impact assessment is required, including for systematic monitoring or other processing likely to create high risk.
- [ ] Map raw and processed data through capture, memory, temporary files, queues, logs, backups, exports, and downstream services.
- [ ] Record foreseeable harms from missed detections, incorrect masking, misuse, re-identification, unauthorized access, and service failure.
- [ ] Define human review, degraded-mode behavior, escalation, and shutdown procedures.
- [ ] Reassess the system when models, cameras, purposes, recipients, or infrastructure change.

### 4. Detection and masking validation

- [ ] Freeze the model file and hash, label mapping, thresholds, input size, masking method, padding, and software version.
- [ ] Evaluate representative footage across relevant lighting, distance, angle, motion, occlusion, demographics, environments, and camera types.
- [ ] Record false negatives separately for each protected class; an aggregate score can hide unsafe failure modes.
- [ ] Inspect processed output for remaining direct and indirect identifiers.
- [ ] Define acceptance criteria and preserve the evaluation dataset, notebook or script, results, and reviewer sign-off.
- [ ] Repeat validation after material configuration or environment changes.

### 5. Security and operations

- [ ] Apply access control, authentication, encryption in transit and at rest, secret management, patching, backup protection, and secure deletion outside this library.
- [ ] Verify actual outbound traffic and dependencies at the deployment boundary; do not infer them from the library description.
- [ ] Keep raw inputs only where necessary, with an approved retention and deletion process covering caches and backups.
- [ ] Avoid placing unnecessary personal data in filenames, operation logs, alerts, or monitoring systems.
- [ ] Test incident detection, containment, recovery, notification assessment, and evidence preservation.
- [ ] Regularly test whether technical and organisational controls remain effective.

### 6. Rights and ongoing governance

- [ ] Provide processes for applicable access, erasure, objection, restriction, rectification, and portability requests.
- [ ] Ensure stored media can be located and acted on without collecting unnecessary new identifiers.
- [ ] Document exceptions and decisions rather than assuming masked output falls outside the GDPR.
- [ ] Train operators on detection limits and prohibit claims that the tool guarantees anonymity or compliance.

## Suggested technical assessment record

Record at least:

- deployment owner, purpose, locations, camera and input specifications;
- model source, file hash, supported classes, label mapping, and software version;
- thresholds, input dimensions, padding, and masking method;
- evaluation dataset scope, conditions, false negatives, false positives, and reviewer;
- raw and processed data flows, external dependencies, storage, logs, recipients, and transfers;
- access controls, encryption, retention, deletion, backup, and incident procedures;
- DPIA or legal-review reference, unresolved risks, approval decision, and next review date.

## Safe public wording

Accurate wording describes the configuration and its limits, for example:

> PrivacyGuard can process frames locally and apply configurable masking to detected regions. Detection failures and contextual identifiers may remain. The component does not by itself establish anonymity or legal compliance.

Avoid claims such as “GDPR compliant,” “anonymous by default,” “zero breach risk,” or “no personal data” unless a qualified, deployment-specific assessment supports the exact statement.

## Official references

- [Regulation (EU) 2016/679 — official consolidated text](https://eur-lex.europa.eu/eli/reg/2016/679/)
- [ICO: How do we ensure anonymisation is effective?](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/data-sharing/anonymisation/how-do-we-ensure-anonymisation-is-effective/)
- [ICO: Pseudonymisation](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/data-sharing/anonymisation/pseudonymisation/)
- [European Data Protection Board documents](https://www.edpb.europa.eu/our-work-tools/documents/our-documents_en)

Recheck applicable law and regulator guidance before relying on this checklist.
