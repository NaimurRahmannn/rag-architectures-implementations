# Authentication Guide

## Overview

The API uses token-based authentication. Every protected request must include a valid access token in the Authorization header.

## AUTH-401

AUTH-401 indicates that authentication credentials are missing, invalid, or expired. Clients should obtain a new access token and retry the request.

## AUTH-403

AUTH-403 indicates that the user is authenticated but does not have permission to access the requested resource.

## AUTH-404

AUTH-404 indicates that the requested resource does not exist.

## Token Expiration

Access tokens expire after a fixed lifetime. When an access token expires, the API may return AUTH-401. Clients should refresh the token or authenticate again.

## Refresh Tokens

Refresh tokens can be used to obtain a new access token without requiring the user to enter credentials again.

## Authorization Header

Protected API requests must include the access token using the Authorization header:

Authorization: Bearer <access_token>

## Security

Access tokens should not be logged or exposed to untrusted clients. Applications should store credentials securely and transmit requests over HTTPS.