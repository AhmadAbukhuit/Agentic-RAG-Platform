# Security Policy

The security of the **Agentic RAG Development Platform** is a top priority. We take security vulnerabilities seriously and appreciate responsible disclosure from the community.

---

## 🛡️ Supported Versions

We release security updates and patches for the following versions:

| Version | Supported          |
| :---    | :---               |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## 🚨 Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability or security defect, please report it through one of the following channels:

1. **GitHub Private Security Advisory (Preferred)**:
   * Go to the **Security** tab of this repository.
   * Click on **Report a vulnerability** to open a private draft advisory.
2. **Email Disclosure**:
   * Send an encrypted or confidential email to `engahmadaukhuit@gmail.com` (or the maintainer's primary contact).
   * Include the subject line: `[SECURITY] Agentic RAG Vulnerability Report`.

### What to Include in Your Report

To help us triage and resolve the issue quickly, please provide:

* Description of the vulnerability and its potential impact.
* Step-by-step instructions or Proof of Concept (PoC) to reproduce the behavior.
* Affected components, nodes, or files (e.g. `app/tools/sql_db.py`, `app/main.py`).
* Any proposed mitigations or code fixes.

### Response & Remediation Timelines

* **Initial Acknowledgment**: Within **48 hours**.
* **Triage & Severity Assessment**: Within **5 business days**.
* **Remediation & Patch Release**: Within **14 business days** depending on complexity.

We will coordinate public disclosure with you after the fix is published.

---

## 🔒 Built-in Security Best Practices

This template adheres to defense-in-depth principles:

### 1. Secrets & Credentials Management

* **Zero Hardcoded Secrets**: All API keys (`LANGCHAIN_API_KEY`, `APP_API_KEY`) and database URIs (`QDRANT_URL`, `REDIS_URL`, `MONGODB_URL`) must be supplied via environment variables (`.env`).
* Git repositories ignore `.env` files automatically via [`.gitignore`](.gitignore).
* Automated secret scanning runs on every push and pull request via TruffleHog.

### 2. SQL Injection Prevention

* The [`app/tools/sql_db.py`](./app/tools/sql_db.py) tool implements strict AST / keyword inspection to block destructive SQL statements (`DROP`, `DELETE`, `TRUNCATE`, `ALTER`, `INSERT`, `UPDATE`), restricting the agent to read-only analytical queries.

### 3. Safe Cross-Origin Resource Sharing (CORS)

* In [`app/main.py`](./app/main.py), wildcard origins (`*`) are automatically decoupled from `allow_credentials=True` to prevent CSRF cross-origin credential leakage.

### 4. Automated Security CI/CD Scans

* **Bandit**: Static AST code analysis scanning for insecure Python calls, dangerous deserialization, and weak cryptography.
* **Pip-Audit**: Continuous CVE scanning of all packages in `requirements.txt` against the PyPI Vulnerability Database.
