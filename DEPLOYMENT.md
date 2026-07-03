# Searchbox Deployment

This project is deployed on the same non-GPU DigitalOcean Droplet as dataroom.

- DigitalOcean project: `AlexandrIA`
- Droplet: `alexandria-apps-fra1`
- Droplet ID: `581992988`
- Region: `fra1`
- Size: `s-4vcpu-8gb`
- Public IPv4: `142.93.173.1`
- Service user: `alexandria`
- App path: `/opt/alexandria/searchbox`
- Service: `alexandria-searchbox.service`
- Local app port: `8001`
- Public hostname prepared in Caddy: `searchbox.tryalexandria.fr`

The deployment is CPU-only and uses the remote Scaleway OpenAI-compatible LLM plus Jina API embedding/reranker settings from `.env`. There is no local LLM inference on the droplet.

## Runtime Layout

The droplet uses:

- A shared Python virtualenv at `/opt/alexandria/venv`.
- A project-local Pi CLI install at `/opt/alexandria/searchbox/.pi-cli`.
- The searchbox Pi compaction patch applied to that project-local Pi install.
- Caddy as the reverse proxy on ports `80` and `443`.
- DigitalOcean Cloud Firewall `alexandria-apps-fra1-fw`, allowing inbound `22`, `80`, and `443`.
- A 4 GB swapfile for dependency installs and heavier Python runtime spikes.

The service command sources `/opt/alexandria/searchbox/.env` through Bash, then forces deployment-specific values:

```bash
PORT=8001
JOBS_DIR=/opt/alexandria/searchbox/data/jobs
EMBED_DEVICE=cpu
HF_HOME=/opt/alexandria/searchbox/data/hf-cache
PI_BIN=/opt/alexandria/searchbox/.pi-cli/node_modules/.bin/pi
```

The deployed Scaleway LLM runtime uses Chat Completions in non-streaming mode. Keep these values in `.env` unless the Scaleway endpoint behavior changes:

```bash
MODEL_API=openai-completions
MAX_OUTPUT_TOKENS=512
THINKING_LEVEL=off
MODEL_REASONING=false
PI_OPENAI_COMPLETIONS_NONSTREAM=1
PI_OPENAI_RESPONSES_NONSTREAM=0
```

The non-streaming compatibility shim is applied to the project-local Pi package by `scripts/pi_ovh_chat_nonstream_patch.py`. The script name is historical; the shim is provider-agnostic for OpenAI-compatible Chat Completions.

The deployed copy intentionally excludes local `data/`, `.venv/`, `.git/`, `node_modules/`, and cache directories. No local job history was copied to the droplet.

## Basic Auth

Caddy has Basic Auth in front of the searchbox hostname. The credentials are not stored in this repository.

Retrieve the current credentials on the droplet:

```bash
ssh root@142.93.173.1 'cat /root/alexandria-basic-auth.txt'
```

Update the password:

```bash
ssh root@142.93.173.1
AUTH_USER=alexandria
AUTH_PASS='replace-with-new-password'
AUTH_HASH=$(caddy hash-password --plaintext "$AUTH_PASS")
printf 'Basic auth for searchbox.tryalexandria.fr and dataroom.tryalexandria.fr\nusername=%s\npassword=%s\n' "$AUTH_USER" "$AUTH_PASS" > /root/alexandria-basic-auth.txt
chmod 600 /root/alexandria-basic-auth.txt
vi /etc/caddy/Caddyfile
caddy fmt --overwrite /etc/caddy/Caddyfile
caddy adapt --config /etc/caddy/Caddyfile >/dev/null
systemctl reload caddy
```

Replace the hash in the `basicauth` block with `$AUTH_HASH`.

## Update Env And Restart

Edit the deployed env file:

```bash
ssh root@142.93.173.1
vi /opt/alexandria/searchbox/.env
systemctl restart alexandria-searchbox
systemctl status alexandria-searchbox --no-pager
```

Check logs:

```bash
journalctl -u alexandria-searchbox -f
```

Check local health from the droplet:

```bash
curl -sS http://127.0.0.1:8001/health
```

Check through Caddy before DNS is configured:

```bash
AUTH="$(awk -F= '/^username=/{u=$2}/^password=/{p=$2}END{print u ":" p}' /root/alexandria-basic-auth.txt)"
curl -u "$AUTH" -H 'Host: searchbox.tryalexandria.fr' http://127.0.0.1/health
curl -k -u "$AUTH" --resolve searchbox.tryalexandria.fr:443:142.93.173.1 https://searchbox.tryalexandria.fr/health
```

## Deploy Code Updates

From the local machine:

```bash
cd /Users/fpa/projects/alexandria
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'node_modules/' \
  --exclude 'data/' \
  --exclude 'models/' \
  --exclude 'out/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.idea/' \
  searchbox/ root@142.93.173.1:/opt/alexandria/searchbox/

ssh root@142.93.173.1 'chown -R alexandria:alexandria /opt/alexandria/searchbox && systemctl restart alexandria-searchbox'
```

If Python dependencies changed:

```bash
ssh root@142.93.173.1 'sudo -u alexandria /opt/alexandria/venv/bin/python -m pip install -r /opt/alexandria/searchbox/server/requirements.txt && systemctl restart alexandria-searchbox'
```

If `PI_VERSION` changes, reinstall the project-local Pi package and reapply both Pi patches:

```bash
ssh root@142.93.173.1 '
  VERSION=$(tr -d "[:space:]" < /opt/alexandria/searchbox/PI_VERSION)
  sudo -u alexandria npm --prefix /opt/alexandria/searchbox/.pi-cli install @earendil-works/pi-coding-agent@$VERSION
  sudo -u alexandria /opt/alexandria/venv/bin/python /opt/alexandria/searchbox/scripts/pi_compaction_patch.py /opt/alexandria/searchbox/.pi-cli/node_modules/@earendil-works/pi-coding-agent/dist/core/agent-session.js
  sudo -u alexandria /opt/alexandria/venv/bin/python /opt/alexandria/searchbox/scripts/pi_ovh_chat_nonstream_patch.py /opt/alexandria/searchbox/.pi-cli/node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai
  node --check /opt/alexandria/searchbox/.pi-cli/node_modules/@earendil-works/pi-coding-agent/dist/core/agent-session.js
  node --check /opt/alexandria/searchbox/.pi-cli/node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js
  systemctl restart alexandria-searchbox
'
```

## Cloudflare Setup Left To Do

In the Cloudflare zone for `tryalexandria.fr`:

1. Add an `A` record:
   - Name: `searchbox`
   - IPv4 address: `142.93.173.1`
   - Proxy status: Proxied is fine.
2. Set SSL/TLS mode to `Full` for immediate end-to-end encryption to the Caddy origin. Caddy currently serves an internal origin certificate on `https://searchbox.tryalexandria.fr`.
3. For `Full (strict)`, either:
   - Install a Cloudflare Origin Certificate on the droplet and change the Caddy `tls` directive to use that cert/key, or
   - After DNS resolves to the droplet, remove `tls internal` from the HTTPS vhost and reload Caddy so it can request a public ACME certificate. If proxied ACME validation fails, temporarily set the DNS record to DNS-only, reload Caddy, then turn proxying back on.
4. Visit `https://searchbox.tryalexandria.fr/health`. Cloudflare should show a trusted browser certificate and Caddy should prompt for Basic Auth.

Do not expose ports `8000` or `8001` in Cloudflare or DigitalOcean; only route through Caddy on `80`/`443`.

## Verification Performed

Health and auth checks passed:

```bash
curl -H 'Host: searchbox.tryalexandria.fr' http://142.93.173.1/health
# 401 without auth

curl -u "$AUTH" -H 'Host: searchbox.tryalexandria.fr' http://142.93.173.1/health
# {"ok": true}

curl -k -u "$AUTH" --resolve searchbox.tryalexandria.fr:443:142.93.173.1 https://searchbox.tryalexandria.fr/health
# {"ok": true}
```

Jina API access from the droplet was verified separately through the dataroom deployment. Scaleway non-streaming Chat Completions was verified from the droplet with the deployed env and returned `HTTP 200`. Direct Scaleway tool-calling was also verified with the selected model.

Searchbox smoke job submitted through Caddy with the Scaleway dataroom zip from dataroom job `38ce7c64d890`:

```text
job_id=26cff28aee4e
terminal_state=done
stop_reason=budget_spent
rc=0
```

`/jobs/26cff28aee4e/answer` returned:

```text
- Scaleway inference API is reachable from this deployment.
- The uploaded dataroom zip was accepted by searchbox.
```
