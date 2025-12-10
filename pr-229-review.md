# PR Review Tasks - PR #229

## PR Information

| Field | Value |
|-------|-------|
| **Title** | Fix Temporal Consistency Bug in reward_forcing v2v Mode |
| **State** | OPEN |
| **Author** | @ryanontheinside |
| **Repository** | daydreamlive/scope |
| **Base Branch** | `main` |
| **Head Branch** | `ryanontheinside/fix/reward-forcing-temporal-fix` |
| **Changes** | +86 -2 across 3 files |
| **URL** | https://github.com/daydreamlive/scope/pull/229 |
| **Created** | Dec 9, 2025, 1:37 PM |
| **Updated** | Dec 9, 2025, 4:40 PM |

### Description

### Problem
The reward_forcing pipeline had a temporal consistency bug in video-to-video (v2v) mode where `current_start_frame` tracked latent frames instead of pixel frames, apparently causing misalignment in temporal tracking.

### Root Cause
The shared `PrepareNextBlock` increments `current_start_frame` by latent frame count, but the VAE temporally expands latents to pixels (e.g., 4 latent frames → 13 pixel frames on first batch, 3 latent frames → 12 pixel frames on subsequent batches).

While `current_start_frame` is used internally by the generator/KV cache in latent space, proper v2v temporal consistency requires tracking the actual number of generated pixel frames for seed generation and state management.

**Note:** longlive may have the same underlying issue but uses `RecacheFramesBlock` which maintains a separate latent buffer that could mask the symptoms.

### Solution
Created reward_forcing-specific `PrepareNextPixelFrameBlock` that:
- Takes `output_video` from `DecodeBlock` as input
- Increments `current_start_frame` by actual pixel frames generated
- Ensures proper temporal tracking aligned with VAE output
- Maintains compatibility with existing generator/KV cache usage

NOTE: A more architectural solution would separate latent vs pixel frame tracking globally, but that requires changes across shared blocks and all pipelines. This could be considered more seriously if we implement another pipeline requiring pixel frame tracking. This pipeline-specific fix achieves the same result with minimal impact.

## Review Summary

| Reviewer | State | Date |
|----------|-------|------|
| @yondonfu | [COMMENTED] | Dec 9, 2025, 4:40 PM |

## Statistics

- **Total Discussion Threads:** 1
- **Total Replies:** 0
- **Files with Comments:** 0

## Files Changed in PR

- `src/scope/core/pipelines/reward_forcing/blocks/__init__.py` (+5 -0)
- `src/scope/core/pipelines/reward_forcing/blocks/prepare_next_pixel.py` (+78 -0)
- `src/scope/core/pipelines/reward_forcing/modular_blocks.py` (+3 -2)

---

## General PR Comments

### Task 1

- [ ] **Status:** Pending

**Reviewer:** @yondonfu
**Review State:** COMMENTED
**Posted:** Dec 9, 2025, 4:40 PM

**Comment:**

I'm not sure I understand these changes so a question...

IIUC `current_start_frame` is supposed to be incremented by the # of latent frames generated because the generator internally is uses the # of latent frames generated thus far multiplied by the # of tokens per latent frame (eg `frame_seq_length`) when indexing into the KV cache.

So, why do we need to all of sudden increment by the # of pixel frames when this is not required for other models? And why was there not a problem with T2V, but only V2V?
