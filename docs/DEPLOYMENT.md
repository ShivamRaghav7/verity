# Bounded local deployment verification

`scripts/verify_deployment.py` checks the structured HTTP workflow on a **fresh,
isolated loopback instance**. It uses only Python's standard library. Start that
instance separately with a new database or dedicated empty Docker volume and
the configured administrator token. Keep the normal workspace database separate.
The script does not start Docker or delete databases. Startup-only health polling
has a default 30-second window and at most 30 attempts. Each attempt's request
timeout is bounded by the remaining window. Failed attempts remain in the report;
failure to become ready produces a refused report without attempting mutation.

Run it only after the isolated server is ready. The token stays in an environment
variable; supply the variable's name rather than its value in the command:

```powershell
.\.venv\Scripts\python.exe scripts/verify_deployment.py --base-url http://127.0.0.1:8001 --admin-token-env EDITGUARD_ADMIN_TOKEN --output reports/deployment-smoke-new.json
```

The server and verification shell must already have the same token configured.
The report excludes authentication headers and redacts the token from strings.
Use a fresh report path for each run. HTTP targets must be literal loopback
addresses or `localhost` (normalized to `127.0.0.1`); credentials, redirects,
remote origins, proxy routing and URL paths are rejected. Before any mutation,
health and history must both report revision 1, with no corrections and only the
initial catalogue-seeding event. The expected initial five-device fact must also
pass. Otherwise the tool refuses the workflow and records unrun phases.

The 23 fixed phases cover the initial answer, draft without authority, approval,
direct/paraphrased correction and provenance, capacity wording, unaffected price,
Pro, older version, similar product and historical date, a conflicting draft,
rejected approval and withheld answer, explicit conflict withdrawal, restored
correction, rollback, restored original behavior, complete audit history, cleared
conflicts and export. Successful rollback retains withdrawn corrections and audit
events: the instance has history afterward and cannot pass the fresh precondition
again. Repetition requires a separately prepared fresh instance.

At most 32 workflow HTTP requests are allowed, separately from the at most 30
startup health attempts. Each request has a finite timeout (default 5 seconds,
configurable from 0.1 to 30) and a bounded 1 MiB response. `--startup-timeout`
sets a readiness window from 0.1 to 30 seconds. Workflow requests are not retried.
Failures retain phase, HTTP/application status and request evidence;
dependent phases remain `not_run`. The pass-rate denominator always contains all
23 planned checks. Exit codes are 0 for all checks passing, 1 for workflow failures
and 2 for refused preconditions. The script attempts withdrawal of its known
corrections even after later assertions fail; inspect failed reports before any
manual recovery. It never withdraws preexisting corrections.

These are deterministic structured engineering checks on the controlled Nimbus
catalogue, with `ml_results=null`. They do not establish learned editing accuracy,
GPU deployment, production authentication or arbitrary-language reliability.
Record the actual build command, image identity and isolated volume separately.
Restart/persistence verification is a separate check after this workflow; the
fresh-instance verifier deliberately refuses a database with preserved history.

After the successful smoke, restart its isolated container and compare the
authoritative facts, corrections, events, dataset identity and revision:

```powershell
docker restart editguard-deployment-c89e528
.\.venv\Scripts\python.exe scripts/verify_persistence.py --base-url http://127.0.0.1:8001 --reference reports/deployment-smoke-c89e528.json --output reports/deployment-restart-new.json
```

The reference must be a successful report from the same loopback origin with both
owned edits withdrawn. The check binds its reference file hash and compares exact
authoritative snapshots, excluding the export's newly measured engineering timing.
It also requires the original five-device structured answer to remain restored.
It makes no administrator mutation and never executes the restart itself. Its
three planned checks all remain in the denominator on errors or missing readiness.

## Recorded local verification

Docker Desktop's Linux engine started successfully on 2026-10-11. The actual image
build and isolated workflow passed: `reports/deployment-smoke-c89e528.json` records
23/23 engineering phases. `reports/deployment-restart-c89e528.json` records 3/3
restart checks, with identical snapshots at revision 7 and both edits withdrawn.
The normal port-8000 workspace database remained at revision 1.

The image used Python 3.12.15 and UID 1000, with a dedicated named SQLite volume
and loopback port 8001. Image/base/source identities are recorded in
`reports/deployment-image-c89e528.json`. The isolated verification container was
stopped afterward; its volume and image remain available. This image contains
the lightweight structured app. Optional local neural inference uses the separate
ML environment described in [ML_SETUP.md](ML_SETUP.md).

GitHub Actions defines a deployment job that builds the image, runs the same
workflow, restarts the container, checks persistence, exports both reports and
stops the isolated fixture. Its remote outcome must be checked independently.
