# Purging runtime data from Git history (DRACO-010)

Older commits published runtime files that were later deleted from the tree.
Deletion does not rewrite history: the blobs remain reachable from the commits
that contained them, from every existing clone, and from GitHub's cached views.

## What is exposed

| Path in history | Contents |
|---|---|
| `app/storage/documents.db` | SQLite registry: filenames, local paths, owner ids, content hashes, error strings |
| `app/storage/uploads/` | 11 personal documents uploaded through the app (CVs, university lab reports, coursework, books) |
| `eval_results/` | Evaluation run outputs (question sets, responses, metrics) |

Removed from the working tree by commits `9ba0eed` and `3c776b8`; still present
in history objects on both `upstream` (AmartyaRanjan/Draco) and `origin`
(harshitsharmaaaa/Draco). Check current exposure any time with
`scripts/check_history_exposure.sh` (read-only, exits non-zero while blobs are
reachable).

## What a purge can and cannot do

- Rewriting one repository's history does **not** clean the other. `upstream`
  holds the same objects; a fork-side purge is undone by the next sync from
  upstream. The rewrite must happen on the repo that owns the canonical
  history (upstream) first, then forks reset onto it.
- Even after a purge and force-push, assume the data was seen: GitHub retains
  unreachable objects for a time and serves cached views; forks and stale
  clones keep their own copies. Rotate or remove anything sensitive and treat
  the personal documents as published.

## Steps (run on the repo that owns the canonical history)

1. **Coordinate.** Everyone must re-clone afterwards; open PRs from old
   branches will need rebasing onto the rewritten history.
2. **Fresh mirror clone:**

   ```
   git clone --mirror https://github.com/OWNER/Draco.git draco-purge
   cd draco-purge
   ```

3. **Rewrite all refs** (this drops remotes by design):

   ```
   pip install git-filter-repo
   git filter-repo --invert-paths --path app/storage --path eval_results
   ```

4. **Force-push the rewritten history:**

   ```
   git remote add origin https://github.com/OWNER/Draco.git
   git push --force --all origin
   git push --force --tags origin
   ```

5. **Ask GitHub to drop the old objects.** Contact GitHub Support with the old
   commit SHAs and request a repository cache purge / garbage collection.
   Forks on GitHub keep their own copies until their maintainers do the same.

6. **Verify.** Fresh clone, then `scripts/check_history_exposure.sh`. Also spot
   check the GitHub web UI for an old commit that contained `app/storage/` —
   cached HTML views can outlive the objects.

## Fork sync after the upstream purge

Once upstream is rewritten: `git fetch upstream && git checkout main && git
reset --hard upstream/main && git push --force origin main`, and re-create any
feature branches from the rewritten `main` (this branch included). Open PRs
referencing old SHAs will show the pre-purge diff history.

## Preventing recurrence

- CI job `draco-core` fails when runtime data or `.env` files become tracked
  again (`.github/workflows/ci.yml`, "Reject tracked runtime data").
- `app/storage/`, `eval_results/`, `documents/` and `.env*` are gitignored;
  keep runtime state out of the worktree when committing.
