# Existing Gmail stdio tool

The worker includes the existing `@shinzolabs/gmail-mcp` **1.7.4** integration.
The package lock comes from cheetan's previously installed npm cache; its resolved
versions and integrity hashes are unchanged. Only the root selector was changed
from `^1.7.4` to `1.7.4`. `npm ci` installs it during the image build, preserving
upstream package license files. Node.js matches the GUI build's 22.23.3 runtime.

The original Mycroft scenario still invokes `npx @shinzolabs/gmail-mcp` over stdio.
Its package is available locally under `/app/node_modules`; npm runs offline in
the worker, so startup never installs it. This introduces no additional MCP
service. Credentials remain in the private worker environment. Deployment checks
initialize the protocol and list tools only; they never send email or create drafts.
