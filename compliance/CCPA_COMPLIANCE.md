# CCPA Deployment Considerations for PrivacyGuard

> **Engineering guidance, not legal advice or certification.** This checklist does not determine whether the California Consumer Privacy Act, as amended by the CPRA, applies to an organization or whether a deployment complies with it. Obtain qualified advice for the actual business, data flows, consumers, and uses.

Last reviewed: 22 September 2026.

## What PrivacyGuard can contribute

PrivacyGuard can be configured to detect selected regions and apply blur, pixelation, or solid masking locally before processed media is sent to another system. Reducing visible identifiers before downstream use may reduce exposure, but it does not decide whether the input or output is personal information, sensitive personal information, biometric information, aggregate consumer information, or deidentified information under California law.

Important limitations include:

- raw frames are still collected and processed by the device;
- missed or incorrect detections leave information unmasked;
- context, audio, timestamps, geolocation, metadata, clothing, gait, household information, or other signals may remain linkable;
- blur and pixelation do not prevent every form of re-identification;
- configuration, model quality, input conditions, storage, sharing, and organizational practices determine the actual risk;
- a local component does not prove that the complete system avoids sale, sharing, disclosure, or external transmission.

Do not assume processed video is legally deidentified. Evaluate the current statutory definition, the technical safeguards, organizational processes, re-identification restrictions, public commitments, and the information reasonably available to the business.

## What the library does not provide

PrivacyGuard does not determine CCPA applicability, create notices at collection or privacy policies, process consumer requests, verify requesters, honor opt-out preference signals, manage sale or sharing choices, limit sensitive-information use, set retention periods, delete backups, manage contracts, or prevent discrimination.

The package also does not supply encryption, identity and access management, network controls, storage governance, or a complete cybersecurity program. Those controls belong to the surrounding deployment.

## Deployment checklist

### 1. Applicability and data inventory

- [ ] Determine whether the organization and processing are subject to the current CCPA and regulations.
- [ ] Identify the business, service providers, contractors, third parties, system owner, and privacy contact.
- [ ] Inventory raw frames, processed media, audio, metadata, logs, identifiers, inferences, and linked records.
- [ ] Document purposes, sources, retention, recipients, sale, sharing, disclosure, and cross-context behavioral advertising uses.
- [ ] Classify each data element under the current legal definitions with qualified advice.

### 2. Notice and purpose controls

- [ ] Provide required notice at or before collection using descriptions that match the actual system.
- [ ] State that raw media is processed and that automated masking can miss regions.
- [ ] Limit collection, use, retention, and sharing to disclosed, compatible, and reasonably necessary purposes.
- [ ] Establish change-control review before introducing new models, classes, cameras, recipients, or purposes.
- [ ] Do not claim that local processing or masking automatically removes CCPA obligations.

### 3. Consumer-rights operations

- [ ] Implement applicable methods for requests to know, access, delete, correct, opt out of sale or sharing, and limit the use or disclosure of sensitive personal information.
- [ ] Verify requesters using a documented process without collecting disproportionate new information.
- [ ] Locate responsive media, metadata, logs, derived data, service-provider copies, archives, and backups.
- [ ] Document exceptions, responses, timing, and downstream instructions.
- [ ] Ensure the organization does not discriminate against consumers for exercising applicable rights.
- [ ] Evaluate and honor applicable opt-out preference signals in the surrounding application; PrivacyGuard does not process them.

### 4. Detection and masking validation

- [ ] Freeze the model file and hash, label mapping, thresholds, input size, masking method, padding, and software version.
- [ ] Test representative footage across expected lighting, angle, distance, motion, occlusion, environments, demographics, and camera types.
- [ ] Record false negatives for every protected class and inspect output for contextual or indirect identifiers.
- [ ] Define acceptable failure rates and the operational response when the system falls outside them.
- [ ] Preserve the evaluation data, method, results, reviewer, approval decision, and review date.
- [ ] Repeat validation after material model, configuration, camera, or environment changes.

### 5. Security, retention, and contracts

- [ ] Apply reasonable security appropriate to the complete processing operation, including access controls, encryption, monitoring, patching, backup protection, and incident response.
- [ ] Keep raw and processed media only for documented periods and implement deletion across primary storage, caches, exports, and backups.
- [ ] Verify network behavior and third-party dependencies at the deployment boundary.
- [ ] Avoid unnecessary personal information in filenames, operation logs, alerts, and analytics.
- [ ] Put required terms and use restrictions into service-provider, contractor, and third-party agreements.
- [ ] Review whether current risk-assessment, cybersecurity-audit, or automated-decisionmaking requirements apply to the business and processing.

### 6. Deidentification assessment

- [ ] Identify all information and reasonably available means that could link processed output to a consumer or household.
- [ ] Evaluate masking strength and remaining identifiers using representative adversarial review.
- [ ] Document technical safeguards, business processes, re-identification prohibitions, contractual controls, and public commitments required by the current legal definition.
- [ ] Continue treating output as personal information unless the organization can substantiate a different classification.
- [ ] Reassess when the data is combined with new datasets or disclosed to new recipients.

## Suggested technical assessment record

Record at least:

- business and system owner, processing purpose, applicable locations, and consumer groups;
- camera and input specifications, model source and hash, class mapping, software version, and configuration;
- evaluation conditions, false negatives, false positives, remaining identifiers, reviewer, and acceptance decision;
- raw and processed data flows, storage, logs, analytics, recipients, sale or sharing analysis, and external dependencies;
- notices, request channels, opt-out mechanisms, contracts, retention, deletion, access controls, and incident procedures;
- unresolved risks, legal-review reference, approver, and next review date.

## Safe public wording

Accurate wording describes the technical behavior without reaching a legal conclusion, for example:

> PrivacyGuard can apply configurable masking to detected regions before downstream use. Detection failures and contextual identifiers may remain. Whether the resulting data is deidentified and whether a deployment complies with the CCPA require a system-specific assessment.

Avoid claims such as “CCPA compliant,” “no personal information,” “no deletion requests required,” “no sale possible,” or “automatic opt-out” unless qualified, current, deployment-specific analysis supports the exact statement.

## Official references

- [California Attorney General: California Consumer Privacy Act](https://oag.ca.gov/privacy/ccpa)
- [California Privacy Protection Agency: Laws and regulations](https://cppa.ca.gov/regulations/)
- [CCPA regulations effective 1 January 2026](https://cppa.ca.gov/regulations/pdf/ccpa_statute_eff_20260101.pdf)
- [California Privacy Protection Agency FAQs](https://cppa.ca.gov/faq)

Recheck the current statute, regulations, thresholds, exemptions, and regulator guidance before relying on this checklist.
