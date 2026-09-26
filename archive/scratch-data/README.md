# Preserved scratch data

Data rescued during the 2025-09 cleanup, kept rather than deleted.

- `infrapulse-stale-demo.db` — a stale copy of `infrapulse.db` (was `junk/infrapulse.db`).
  The live database is now empty, so this is the only remaining record of 4 demo
  complaints filed on 2025-09-02. Kept for reference; the app does not read it.

  To restore as the live database (overwrites current data):
  ```bash
  cp archive/scratch-data/infrapulse-stale-demo.db infrapulse.db
  ```
