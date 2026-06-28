You can add a requirement like the following to your condensation prompt.

---

### Multi-sentence segmentation

If the input contains **more than one complete sentence**, split the output into multiple timed segments instead of keeping everything in a single block.

Rules:

1. **Preserve the original sentence order.** Never reorder or merge ideas.

2. **Create new time windows only inside the original window.**

   * The first segment starts at the original start time.
   * The last segment ends at the original end time.
   * Intermediate boundaries are generated proportionally to the amount of text in each segment.
   * Do not extend beyond the original timestamps.

3. **Split only at natural linguistic boundaries.**
   Prefer:

   * sentence endings (`.`, `?`, `!`)
   * major clause boundaries
   * discourse transitions (e.g. "However", "Then", "After that")

   Avoid splitting:

   * noun phrases ("the sacred spring")
   * verb-object pairs ("prepare the sacred salt")
   * fixed expressions
   * names and titles

4. **Balance the segments.**
   Allocate time approximately proportional to the number of characters (or expected speaking duration) in each segment while preserving natural prosody. The segments do **not** need to have identical durations.

5. **Reading speed constraint.**
   Target approximately:

   * **15 characters per second**
   * **2.5 words per second**

   If necessary, redistribute the available time among the generated segments while remaining inside the original time window.

6. **Condense each segment independently.**
   After splitting, perform condensation separately for each segment so each one satisfies the reading-speed target without depending on adjacent segments.

7. **Output format**

```
[start_time - end_time] Condensed sentence 1.

[start_time - end_time] Condensed sentence 2.

...
```

### Example

Input:

```
[265.31s - 290.64s]
She says the virgins carry water from the sacred spring every morning and prepare the sacred salt used in every sacrifice in Rome. She proudly says only virgins may do this. You'll never forget how long you'll serve.
```

Output:

```
[265.31s - 279.10s]
She says the virgins fetch water from the sacred spring every morning and prepare the sacred salt used in Rome's sacrifices.

[279.10s - 290.64s]
She proudly says only virgins may do this. You'll never forget how long you'll serve.
```

This feature should prioritize **natural speech rhythm** over equal-length chunks, while ensuring every generated segment remains within the original subtitle time window and satisfies the target reading speed.
