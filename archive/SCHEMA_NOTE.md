# schema.sql — stale, do not use

`schema.sql` (now `schema.sql.stale`) is an **out-of-date hand-written DDL** that is not used
by the application. The app creates its schema automatically from the SQLAlchemy models in
`app/models.py` via `init_db()` in `app/database.py` (called on startup from the FastAPI
lifespan handler).

It is out of date in ways that would break things if you ran it against a real database:

- It creates a table named `staff`; the app's model is `staff_members`. Running the DDL
  would leave the app querying a table that does not exist.
- It is missing the auto-migration columns that `app/database.py` adds for existing
  complaint rows (`user_id`, `user_email`, `user_phone`, `assigned_staff_id`,
  `assigned_staff_name`).

To reset the database, use the supported script at the repo root:

```bash
python3 reset_db.py
```

`app/models.py` is the single source of truth for the schema.
