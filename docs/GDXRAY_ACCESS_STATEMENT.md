# GDXray Welds Data Access Statement

**Manuscript:** WeldCascade (Sensors, MDPI)  
**Purpose:** Factual log for editors/reviewers when cross-dataset validation is requested but GDXray Welds could not be obtained.  
**Date:** 2026-06-13  

---

## Dataset cited

- **Name:** GDXray — group **Welds** (88 radiographs, ~209 MB ZIP)  
- **Reference:** Mery, D.; *et al.* GDXray: The database of X-ray images for nondestructive testing. *J. Nondestructive Eval.* 2015, 34(4). DOI: [10.1007/s10921-015-0315-7](https://doi.org/10.1007/s10921-015-0315-7)  
- **Official index:** https://domingomery.ing.uc.cl/material/gdxray/  
- **Planned use in WeldCascade:** zero-shot binary defect screening (Steel Pipe–trained YOLOv8n weights; no fine-tuning on GDXray)

---

## Access attempts

| # | Method | URL / path | Result |
|---|--------|------------|--------|
| 1 | Official web page | https://domingomery.ing.uc.cl/material/gdxray/ | Page active; Welds download via Dropbox link |
| 2 | GitHub README mirror | https://github.com/computervision-xray-testing/GDXray | Same Dropbox `Welds.zip` link |
| 3 | Automated HTTP download | `https://www.dropbox.com/scl/fi/im896nbhllnbnol585fsq/Welds.zip?rlkey=u584im2jtrdxzmhrg2lcqtavv&dl=1` | HTTP 200; **`Content-Type: text/html`**; body is HTML interstitial, not ZIP |
| 4 | Legacy direct URL (cited in third-party code) | `http://dmery.sitios.ing.uc.cl/images/GDXray/Welds.zip` | 301 → malformed `https://domingomery.ing.uc.climages/GDXray/Welds.zip`; **404** on corrected URL |
| 5 | Browser manual download | Same Dropbox link | Did not produce verifiable ~209 MB ZIP (HTML / blocked download in author environment) |

**Validation rule used in project:** valid ZIP must begin with magic bytes `PK` (0x50 0x4B) and be >> 1 MB. HTML responses fail this check.

---

## Project artefacts

| File | Status |
|------|--------|
| `scripts/run_gdxray_validation.ps1` | Documents manual-download workflow; rejects non-ZIP files |
| `deliverables/results/sensors_extras.json` | `"gdxray_external": {"available": false}` |
| Manuscript **Limitations (4)** | States GDXray could not be obtained under reproducible download conditions |

---

## Manuscript policy

- **No GDXray metrics** are reported in Abstract, Highlights, Results, or Conclusion.  
- **Limitations item (4)** is retained without alteration.  
- **Future work** mentions cross-dataset evaluation when archives become locally available.  
- **Revision offer:** if a working mirror is provided during review, authors will add zero-shot binary screening results in one revision pass.

---

## Suggested reviewer-facing sentence (English)

> We attempted to download GDXray Welds from the official index, the GitHub mirror, and legacy UC-hosted URLs. Dropbox links returned HTML rather than ZIP under automated retrieval; the legacy direct URL redirects to a broken path (404). We therefore scoped the paper as a single-corpus Steel Pipe benchmark and document this limitation in §Limitations (4), without reporting external validation metrics we did not obtain.
