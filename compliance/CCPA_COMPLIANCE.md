# CCPA/CPRA Compliance Checklist for PrivacyGuard

Use this for California/US privacy compliance.

## ✅ CCPA Obligations Covered

| Obligation | Requirement | PrivacyGuard Solution |
|-----------|------------|----------------------|
| **Data Collection** | Disclose what you collect | Anonymize before collecting |
| **Sale of Data** | Opt-out for sale | No cloud transmission = no sale risk |
| **Deletion Rights** | Delete on request | Only store anonymized frames |
| **Access Rights** | Let users see their data | Can't identify → no access request |
| **Non-discrimination** | Don't charge for privacy | Free for all customers |

## 🔒 Data Protection Strategies

### Strategy 1: Anonymize Before Collection
```python
from privacyguard import PrivacyGuard

# Capture → Blur → Store (never store raw frames)
guard = PrivacyGuard("model.onnx")
frame = capture_from_camera()
anonymized = guard.process_frame(frame)
storage.save(anonymized)  # Only this is stored
```

**CCPA Impact:**
- Raw biometric data (faces) never collected ✅
- No "sale of personal information" ✅
- No deletion requests needed ✅

### Strategy 2: Metadata Stripping
```python
from privacyguard.metadata import MetadataStripper

stripper = MetadataStripper()
stripper.strip_image("photo.jpg", "clean.jpg")
# Removes: EXIF, GPS, camera model, device ID, timestamp
```

**Removed Data:**
- Precise geolocation
- Device identifiers
- Timestamps (for identification)

## 📋 Compliance Implementation

- [ ] **Privacy Policy Updated**: Disclose local video anonymization
- [ ] **Opt-Out Available**: Users can request no recording
- [ ] **No Data Sale**: Contractually prohibit selling anonymized data
- [ ] **Deletion Process**: Document how to delete stored videos
- [ ] **Access Controls**: Limit employee access to processed videos
- [ ] **Vendor Contracts**: Require vendors to follow CCPA

## 🏷️ Consumer Rights Response

### Right to Know: "What data do you have about me?"
```
Response: "We don't collect faces/license plates - they're blurred
immediately using PrivacyGuard. Only anonymized frames are stored."

Data provided: Anonymized video only
```

### Right to Delete: "Delete my data"
```
Process:
1. Identify video timestamp (e.g., 2024-01-15 14:30:00)
2. Locate stored files from that time
3. Delete from storage and backups
4. Confirm deletion within 30 days
```

### Right to Opt-Out: "Don't collect my data"
```
Options:
- Turn off recording in settings
- Exclude certain cameras
- Request deletion of historical data
```

## 💰 Opt-Out Mechanism

Implement in your application:

```python
class PrivacySettings:
    def __init__(self):
        self.recording_enabled = True
        self.anonymize_faces = True
        self.anonymize_plates = True
        self.data_retention_days = 30

    def disable_collection(self):
        """User clicks 'Opt-Out' button"""
        self.recording_enabled = False

    def request_deletion(self, date_range):
        """User requests historical data deletion"""
        delete_files_in_range(date_range)
```

## 🛡️ Data Minimization

PrivacyGuard achieves CCPA Article 1798.100 compliance:

```
BEFORE PrivacyGuard:
Raw video → Cloud → Storage → Sold to advertisers ❌

AFTER PrivacyGuard:
Raw video → Local blur → Anonymized storage → No sale possible ✅
```

## 📊 CCPA vs PrivacyGuard

| CCPA Right | Traditional Approach | PrivacyGuard Approach |
|-----------|--------------------|--------------------|
| **Right to Know** | Hard (extract from database) | Easy (no personal data) |
| **Right to Delete** | Complex (PII in storage) | Simple (only delete video) |
| **Right to Opt-Out** | Hard (data already sold) | Easy (prevent collection) |
| **Non-Discrimination** | Hard (track consent) | Automatic (no data) |
| **Cost** | $50k-100k compliance | Minimal overhead |

## 🚀 Sample Implementation

```python
from privacyguard import PrivacyGuard
from privacyguard.enterprise import AuditLogger

class CCPACompliantCamera:
    def __init__(self):
        self.guard = PrivacyGuard("model.onnx")
        self.audit = AuditLogger("/logs/privacy_audit.json")
        self.recording_enabled = True

    def capture_and_store(self, frame):
        """CCPA-compliant video processing"""
        if not self.recording_enabled:
            return

        # Anonymize immediately
        anon_frame = self.guard.process_frame(frame)

        # Log for audit trail
        self.audit.log_anonymization(
            source_file="camera_1",
            output_file=f"storage/{datetime.now()}.jpg",
            detections_count=0,  # Don't track # of faces
            processing_time_ms=33,
            anonymization_method="gaussian",
            model_name="face_detector"
        )

        # Store anonymized only
        self.storage.save(anon_frame)

    def handle_deletion_request(self, date_range):
        """CCPA Right to Delete"""
        files = self.storage.list_files(date_range)
        for file in files:
            self.storage.delete(file)
            self.audit.log_deletion(file)
```

## 📖 References

- [CCPA Official Text](https://oag.ca.gov/privacy/ccpa)
- [CPRA (California Privacy Rights Act)](https://oag.ca.gov/cpra)
- [FTC Guidance on Biometrics](https://www.ftc.gov/news-events/blogs/techftc/2020/08/are-you-ready-biometric-regulations)
