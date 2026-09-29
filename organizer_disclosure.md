# Disclosure: blinded test compounds present in the public Octant CYP release

Subject: Possible test-set overlap in Octant_CYP_inhibition_reactivity_blog_release

Hello,

While running a pre-ingest leakage check on the Octant CYP release
(`openadmet/Octant_CYP_inhibition_reactivity_blog_release`) before considering it as
auxiliary data, we found that some blinded test compounds appear in that public release.

Specifically, of the 750 blinded regression test compounds:

- **5 match by exact structure** (full InChIKey) **and by matching OCNT core identifier**,
  with `CYP3A4_pIC50` values attached in the release:
  - OCNT-0493952
  - OCNT-2308485
  - OCNT-2311186
  - OCNT-2312792
  - OCNT-2314689
- **1 additional near-duplicate** at Tanimoto 0.952:
  - OCNT-2534939

We quarantined all 6 compounds from any downstream use and did **not** inspect the matched
assay values. We only read structure and identifier columns during the check.

Flagging this in case it is useful for the challenge. We are happy to share the quarantine
list if that helps.

Best regards,
