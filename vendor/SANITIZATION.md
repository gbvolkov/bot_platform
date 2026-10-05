# Source fixture sanitization

Source revision: `31d7a16964ba68bae6e00650c96d32538b1fe764`.

GitHub push protection identified a personal-access-token-shaped value in two
upstream fixture files. Its validity was not checked. The original value has
been removed from this branch's source and bundled archive.

- `tests/input_scanners/test_secrets.py` constructs an explicitly synthetic
  all-zero token to retain GitHub token detection coverage.
- `benchmarks/input_examples.json` uses the public AWS documentation example
  already present in the dependency's tests for its secrets-scanning input.

The archive contains the same sanitized fixtures as the expanded source.
Runtime modules, package metadata, and the original license are unchanged.
