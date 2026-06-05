### Implementation Details

- Keep assertion checks simple `assert foo, bar`, don't use ifs
- Avoid default values other than in test code
- Keep everything except test deterministic by explicitly asking for a seed
