# CONN-GITHUB-AGENT-TRANSPORT-TLS — Transport trust comes from the SDK TLS profile

Owner: github-agent. Cross-repo IDs: `SDK-CONNECTOR-CONTROL-R017`, `SDK-CENTRALIZED-CAPABILITIES-R008`.

## Purpose and user stories

An operator points this connector at a private CA, or at GitHub Enterprise behind one, through the shared TLS profile settings. Every HTTPS call the connector makes, the GraphQL client and the fixed-origin skill-script client alike, verifies the peer with that one profile.

## Requirements and acceptance

See [requirements.md](requirements.md).

## Why a new spec

`specs/connector-identity` owns only the fleet name. No other local spec owns transport or trust.
