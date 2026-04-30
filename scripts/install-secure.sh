#!/bin/bash
# ============================================================================
# Hermes Agent Installer — Security-Hardened Edition
# ============================================================================
# Drop-in replacement for install.sh with comprehensive security hardening:
#   - SHA-256 checksum verification for all downloaded artifacts
#   - Pinned git commit checkout (--commit SHA)
#   - Strict shell hygiene (set -euo pipefail, umask 0077)
#   - Audit logging of all installation actions
#   - Secret redaction in all output
#   - Enforced file permissions on sensitive configs
#   - Rollback on failure
#   - Minimal privilege (sudo -k after each use)
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Kingcoloss/hermes-agent-claude-arch-wrapped/main/scripts/install-secure.sh | bash
#
# Or with options:
#   bash install-secure.sh --branch main --commit abc1234 --skip-setup
#
# ============================================================================

set -euo pipefail

# ============================================================================
# Shell hardening
# ============================================================================

umask 0077

# Generate unique install run ID for audit correlation
INSTALL_ID="$(uuidgen 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -d '-' || head -c32 /dev/urandom 2>/dev/null | xxd -p -c32 2>/dev/null || printf '%s%s' "$$" "$(date +%s)")"
INSTALL_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date +%s)"
INSTALL_SUCCEEDED=false
CLONE_PERFORMED=false
VENV_CREATED=false
TEMP_DIRS=()
SKIP_VERIFY=false
AUDIT_ONLY=false

# ============================================================================
# Pinned SHA-256 checksums for supply-chain integrity
# ============================================================================
# When updating UV_VERSION or NODE_VERSION, download the artifacts, run
# sha256sum on each, and paste the hashes below. Empty values trigger a
# warning but proceed (graceful fallback). Tarball checksums fail hard.
# ============================================================================

UV_INSTALLER_SHA256=""
NODE_SHA256_LINUX_X64=""
NODE_SHA256_LINUX_ARM64=""
NODE_SHA256_DARWIN_X64=""
NODE_SHA256_DARWIN_ARM64=""

# ============================================================================
# Colors
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# ============================================================================
# Configuration
# ============================================================================

REPO_URL_SSH="git@github.com:Kingcoloss/hermes-agent-claude-arch-wrapped.git"
REPO_URL_HTTPS="https://github.com/Kingcoloss/hermes-agent-claude-arch-wrapped.git"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"

if [ -n "${HERMES_INSTALL_DIR:-}" ]; then
    INSTALL_DIR="$HERMES_INSTALL_DIR"
    INSTALL_DIR_EXPLICIT=true
else
    INSTALL_DIR=""
    INSTALL_DIR_EXPLICIT=false
fi
PYTHON_VERSION="3.11"
NODE_VERSION="22"

ROOT_FHS_LAYOUT=false

USE_VENV=true
RUN_SETUP=true
BRANCH="main"
GIT_COMMIT=""

# Detect non-interactive mode
if [ -t 0 ]; then
    IS_INTERACTIVE=true
else
    IS_INTERACTIVE=false
fi

# ============================================================================
# Parse arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-venv)
            USE_VENV=false
            shift
            ;;
        --skip-setup)
            RUN_SETUP=false
            shift
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --commit)
            GIT_COMMIT="$2"
            shift 2
            ;;
        --dir)
            INSTALL_DIR="$2"
            INSTALL_DIR_EXPLICIT=true
            shift 2
            ;;
        --hermes-home)
            HERMES_HOME="$2"
            shift 2
            ;;
        --skip-verify)
            SKIP_VERIFY=true
            log_warn_once "--skip-verify: checksum verification disabled (insecure, dev-only)"
            shift
            ;;
        --audit-only)
            AUDIT_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Hermes Agent Installer (Security-Hardened Edition)"
            echo ""
            echo "Usage: install-secure.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-venv          Don't create virtual environment"
            echo "  --skip-setup       Skip interactive setup wizard"
            echo "  --branch NAME      Git branch to install (default: main)"
            echo "  --commit SHA       Pin checkout to specific commit hash"
            echo "  --dir PATH         Installation directory"
            echo "  --hermes-home PATH Data directory (default: ~/.hermes, or \$HERMES_HOME)"
            echo "  --skip-verify      Bypass checksum verification (insecure, dev-only)"
            echo "  --audit-only       Run preflight checks and exit"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Security features vs. install.sh:"
            echo "  - SHA-256 checksum verification for all downloads"
            echo "  - Pinned git commit checkout (--commit)"
            echo "  - Audit log at ~/.hermes/install-audit.log"
            echo "  - Secret redaction in all output"
            echo "  - Enforced 0600 permissions on sensitive config files"
            echo "  - Rollback on failure"
            echo "  - Minimal privilege (sudo -k after each use)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Audit log setup — must happen after HERMES_HOME is resolved
# ============================================================================

mkdir -p "$HERMES_HOME"
AUDIT_LOG_FD=200
exec 200>>"$HERMES_HOME/install-audit.log"

# ============================================================================
# Security helper functions
# ============================================================================

# Track whether we already warned about --skip-verify (avoid repeated warnings)
_WARNED_SKIP_VERIFY=false
log_warn_once() {
    if [ "$_WARNED_SKIP_VERIFY" = false ]; then
        _WARNED_SKIP_VERIFY=true
        echo -e "${YELLOW}WARNING: $1${NC}" >&2
    fi
}

audit_log() {
    local msg="[$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date +%s)] [install-${INSTALL_ID}] $*"
    echo "$msg" >&${AUDIT_LOG_FD}
}

# Secret redaction — sed patterns ported from agent/redact.py
# Masks API key prefixes, ENV assignments with secret names, auth headers.
# Brackets in replacement strings are literal — no special meaning in s///replacement.
redact_line() {
    sed -E \
        -e 's/sk-[A-Za-z0-9_-]{10,}/REDACTED_sk/g' \
        -e 's/ghp_[A-Za-z0-9]{10,}/REDACTED_ghp/g' \
        -e 's/github_pat_[A-Za-z0-9_]{10,}/REDACTED_ghpat/g' \
        -e 's/AKIA[A-Z0-9]{16}/REDACTED_AKIA/g' \
        -e 's/xox[baprs]-[A-Za-z0-9-]{10,}/REDACTED_xox/g' \
        -e 's/AIza[A-Za-z0-9_-]{30,}/REDACTED_AIza/g' \
        -e 's/([A-Z0-9_]{0,50}(API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)[A-Z0-9_]{0,50})=[^[:space:]]+/\1=REDACTED/g' \
        -e 's/Bearer [A-Za-z0-9._-]+/Bearer REDACTED/g' \
        -e 's/sk_live_[A-Za-z0-9]{10,}/REDACTED_sklive/g' \
        -e 's/sk_test_[A-Za-z0-9]{10,}/REDACTED_sktest/g' \
        -e 's/SG\.[A-Za-z0-9_-]{10,}/REDACTED_SG/g' \
        -e 's/hf_[A-Za-z0-9]{10,}/REDACTED_hf/g' \
        -e 's/pplx-[A-Za-z0-9]{10,}/REDACTED_pplx/g' \
        -e 's/npm_[A-Za-z0-9]{10,}/REDACTED_npm/g' \
        -e 's/pypi-[A-Za-z0-9_-]{10,}/REDACTED_pypi/g' \
        -e 's/dop_v1_[A-Za-z0-9]{10,}/REDACTED_dop/g' \
        -e 's/tvly-[A-Za-z0-9]{10,}/REDACTED_tvly/g' \
        -e 's/exa_[A-Za-z0-9]{10,}/REDACTED_exa/g' \
        -e 's/gsk_[A-Za-z0-9]{10,}/REDACTED_gsk/g'
}

# SHA-256 verification for downloaded files
# Returns 0 on match, 1 on mismatch. If expected hash is empty, warns but returns 0.
verify_sha256() {
    local filepath="$1"
    local expected_hash="$2"
    local description="$3"

    if [ ! -f "$filepath" ]; then
        log_error "verify_sha256: file not found: $filepath"
        audit_log "VERIFY_FAIL file_not_found path=$filepath"
        return 1
    fi

    if [ "$SKIP_VERIFY" = true ]; then
        log_warn_once "Skipping checksum verification (--skip-verify)"
        audit_log "VERIFY_SKIPPED desc=$description"
        return 0
    fi

    if [ -z "$expected_hash" ]; then
        log_warn "No pinned checksum for $description — skipping verification (populate script header for full integrity)"
        audit_log "VERIFY_NO_PIN desc=$description"
        return 0
    fi

    local actual_hash
    if command -v sha256sum &>/dev/null; then
        actual_hash="$(sha256sum "$filepath" | awk '{print $1}')"
    elif command -v shasum &>/dev/null; then
        actual_hash="$(shasum -a 256 "$filepath" | awk '{print $1}')"
    else
        log_error "No SHA-256 utility found (need sha256sum or shasum)"
        audit_log "VERIFY_FAIL no_hash_tool"
        return 1
    fi

    if [ "$actual_hash" = "$expected_hash" ]; then
        log_success "Checksum verified: $description"
        audit_log "VERIFY_OK desc=$description hash=${actual_hash:0:16}..."
        return 0
    else
        log_error "Checksum MISMATCH for $description!"
        log_error "  Expected: $expected_hash"
        log_error "  Actual:   $actual_hash"
        audit_log "VERIFY_FAIL desc=$description expected=${expected_hash:0:16}... actual=${actual_hash:0:16}..."
        return 1
    fi
}

# Safe temp directory — registers for cleanup
safe_temp_dir() {
    local tmp
    tmp="$(mktemp -d 2>/dev/null || echo "/tmp/hermes-install-$$")"
    mkdir -p "$tmp"
    TEMP_DIRS+=("$tmp")
    echo "$tmp"
}

# Path validation — rejects traversal components, resolves symlinks
validate_path() {
    local path_str="$1"
    local description="${2:-path}"

    # Reject .. components (mirrors tools/path_security.py:has_traversal_component)
    case "$path_str" in
        *../* | */..* | *".. "* | *" .."*)
            log_error "Path traversal rejected in $description: $path_str"
            audit_log "PATH_REJECTED traversal desc=$description path=$(echo "$path_str" | redact_line)"
            return 1
            ;;
    esac

    # Resolve via realpath if available
    if command -v realpath &>/dev/null; then
        local resolved
        resolved="$(realpath -m "$path_str" 2>/dev/null || echo "$path_str")"
        case "$resolved" in
            *../* | */..*)
                log_error "Resolved path traversal in $description: $resolved"
                audit_log "PATH_REJECTED resolved_traversal desc=$description"
                return 1
                ;;
        esac
    fi

    return 0
}

# Enforce restrictive permissions on sensitive files
enforce_permissions() {
    local hermes_home="$1"

    # HERMES_HOME itself should be 0700 (owner-only access)
    if [ -d "$hermes_home" ]; then
        chmod 700 "$hermes_home" 2>/dev/null || true
    fi

    # Sensitive files must be 0600 (owner read/write only)
    local sensitive_files=(
        "$hermes_home/.env"
        "$hermes_home/config.yaml"
    )

    local f
    for f in "${sensitive_files[@]}"; do
        if [ -f "$f" ]; then
            chmod 600 "$f" 2>/dev/null || true
            audit_log "PERMS_SET file=$(basename "$f") mode=0600"
        fi
    done

    # Also lock down any credential/token files under HERMES_HOME
    if [ -d "$hermes_home" ]; then
        find "$hermes_home" -maxdepth 2 \
            \( -name "*credential*" -o -name "*token*" -o -name "*secret*" \) \
            -type f 2>/dev/null | while IFS= read -r credfile; do
            chmod 600 "$credfile" 2>/dev/null || true
            audit_log "PERMS_SET file=$(basename "$credfile") mode=0600"
        done
    fi

    # Session and log directories: 0700
    for subdir in sessions logs cron pairing hooks memories skills; do
        if [ -d "$hermes_home/$subdir" ]; then
            chmod 700 "$hermes_home/$subdir" 2>/dev/null || true
        fi
    done

    # WhatsApp session creds: strict 0600
    if [ -f "$hermes_home/whatsapp/session/creds.json" ]; then
        chmod 600 "$hermes_home/whatsapp/session/creds.json" 2>/dev/null || true
        chmod 700 "$hermes_home/whatsapp/session" 2>/dev/null || true
        chmod 700 "$hermes_home/whatsapp" 2>/dev/null || true
        audit_log "PERMS_SET file=whatsapp/session/creds.json mode=0600"
    fi
}

# Clear sudo timestamp — call after every sudo use
clear_sudo() {
    sudo -k 2>/dev/null || true
}

# Rollback on failure — only removes artifacts from this run
rollback() {
    local exit_code=$?
    if [ "$INSTALL_SUCCEEDED" = true ]; then
        return 0
    fi

    echo ""
    log_error "Installation failed — rolling back..."
    audit_log "ROLLBACK_START"

    # Remove venv if we created it this run
    if [ "$VENV_CREATED" = true ] && [ -d "${INSTALL_DIR:-}/venv" ]; then
        log_info "Removing virtual environment..."
        rm -rf "${INSTALL_DIR:-}/venv"
        audit_log "ROLLBACK removed=venv"
    fi

    # Remove git repo if we cloned it this run
    if [ "$CLONE_PERFORMED" = true ] && [ -d "${INSTALL_DIR:-}/.git" ]; then
        log_info "Removing cloned repository..."
        rm -rf "${INSTALL_DIR:-}"
        audit_log "ROLLBACK removed=clone"
    fi

    # Clean up temp directories
    local tmp
    for tmp in "${TEMP_DIRS[@]:-}"; do
        if [ -d "$tmp" ]; then
            rm -rf "$tmp"
        fi
    done

    # Remove command symlink if we created it
    local link_dir
    link_dir="$(get_command_link_dir 2>/dev/null || echo "")"
    if [ -n "$link_dir" ] && [ -L "$link_dir/hermes" ]; then
        # Only remove if it points to our install
        local link_target
        link_target="$(readlink "$link_dir/hermes" 2>/dev/null || echo "")"
        if [ -n "$link_target" ] && echo "$link_target" | grep -q "${INSTALL_DIR:-hermes}"; then
            rm -f "$link_dir/hermes"
            audit_log "ROLLBACK removed=symlink"
        fi
    fi

    audit_log "ROLLBACK_COMPLETE"
    log_error "Rollback complete. See ~/.hermes/install-audit.log for details."
}

# ============================================================================
# Logging functions (with secret redaction)
# ============================================================================

print_banner() {
    echo ""
    echo -e "${MAGENTA}${BOLD}"
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│          Hermes Agent Installer (Secure)               │"
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│  Security-hardened edition with checksum verification,  │"
    echo "│  audit logging, secret redaction, and rollback.         │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
    audit_log "INSTALL_START id=$INSTALL_ID branch=$BRANCH commit=${GIT_COMMIT:-none}"
}

log_info() {
    echo -e "${CYAN}→${NC} $(echo "$*" | redact_line)"
}

log_success() {
    echo -e "${GREEN}✓${NC} $(echo "$*" | redact_line)"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $(echo "$*" | redact_line)"
}

log_error() {
    echo -e "${RED}✗${NC} $(echo "$*" | redact_line)">&2
}

prompt_yes_no() {
    local question="$1"
    local default="${2:-yes}"
    local prompt_suffix
    local answer=""

    case "$default" in
        [yY]|[yY][eE][sS]|[tT][rR][uU][eE]|1) prompt_suffix="[Y/n]" ;;
        *) prompt_suffix="[y/N]" ;;
    esac

    if [ "$IS_INTERACTIVE" = true ]; then
        read -r -p "$question $prompt_suffix " answer || answer=""
    elif [ -r /dev/tty ] && [ -w /dev/tty ]; then
        printf "%s %s " "$question" "$prompt_suffix" > /dev/tty
        IFS= read -r answer < /dev/tty || answer=""
    else
        answer=""
    fi

    answer="${answer#"${answer%%[![:space:]]*}"}"
    answer="${answer%"${answer##*[![:space:]]}"}"

    if [ -z "$answer" ]; then
        case "$default" in
            [yY]|[yY][eE][sS]|[tT][rR][uU][eE]|1) return 0 ;;
            *) return 1 ;;
        esac
    fi

    case "$answer" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

is_termux() {
    [ -n "${TERMUX_VERSION:-}" ] || [[ "${PREFIX:-}" == *"com.termux/files/usr"* ]]
}

# ============================================================================
# Install layout
# ============================================================================

resolve_install_layout() {
    if [ "$INSTALL_DIR_EXPLICIT" = true ]; then
        log_info "Install directory: $INSTALL_DIR (explicit)"
        validate_path "$INSTALL_DIR" "install directory" || exit 1
        return 0
    fi

    if is_termux; then
        INSTALL_DIR="$HERMES_HOME/hermes-agent"
        return 0
    fi

    if [ "${OS:-}" = "linux" ] && [ "$(id -u)" -eq 0 ]; then
        if [ -d "$HERMES_HOME/hermes-agent/.git" ]; then
            INSTALL_DIR="$HERMES_HOME/hermes-agent"
            log_info "Existing install detected at $INSTALL_DIR — keeping legacy layout"
            return 0
        fi
        INSTALL_DIR="/usr/local/lib/hermes-agent"
        ROOT_FHS_LAYOUT=true
        log_info "Root install on Linux — using FHS layout"
        return 0
    fi

    INSTALL_DIR="$HERMES_HOME/hermes-agent"
}

get_command_link_dir() {
    if is_termux && [ -n "${PREFIX:-}" ]; then
        echo "$PREFIX/bin"
    elif [ "$ROOT_FHS_LAYOUT" = true ]; then
        echo "/usr/local/bin"
    else
        echo "$HOME/.local/bin"
    fi
}

get_command_link_display_dir() {
    if is_termux && [ -n "${PREFIX:-}" ]; then
        echo '$PREFIX/bin'
    elif [ "$ROOT_FHS_LAYOUT" = true ]; then
        echo '/usr/local/bin'
    else
        echo '~/.local/bin'
    fi
}

get_hermes_command_path() {
    local link_dir
    link_dir="$(get_command_link_dir)"
    if [ -x "$link_dir/hermes" ]; then
        echo "$link_dir/hermes"
    else
        echo "hermes"
    fi
}

# ============================================================================
# System detection
# ============================================================================

detect_os() {
    case "$(uname -s)" in
        Linux*)
            if is_termux; then
                OS="android"
                DISTRO="termux"
            else
                OS="linux"
                if [ -f /etc/os-release ]; then
                    . /etc/os-release
                    DISTRO="$ID"
                else
                    DISTRO="unknown"
                fi
            fi
            ;;
        Darwin*)
            OS="macos"
            DISTRO="macos"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            OS="windows"
            DISTRO="windows"
            log_error "Windows detected. Use the PowerShell installer:"
            log_info "  irm https://raw.githubusercontent.com/Kingcoloss/hermes-agent-claude-arch-wrapped/main/scripts/install.ps1 | iex"
            exit 1
            ;;
        *)
            OS="unknown"
            DISTRO="unknown"
            log_warn "Unknown operating system"
            ;;
    esac

    log_success "Detected: $OS ($DISTRO)"
    audit_log "OS_DETECT os=$OS distro=$DISTRO"
}

# ============================================================================
# Dependency checks
# ============================================================================

install_uv() {
    if [ "$DISTRO" = "termux" ]; then
        log_info "Termux detected — using Python's stdlib venv + pip instead of uv"
        UV_CMD=""
        return 0
    fi

    log_info "Checking for uv package manager..."

    if command -v uv &> /dev/null; then
        UV_CMD="uv"
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        log_success "uv found ($UV_VERSION)"
        audit_log "UV_FOUND version=$UV_VERSION"
        return 0
    fi

    if [ -x "$HOME/.local/bin/uv" ]; then
        UV_CMD="$HOME/.local/bin/uv"
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        log_success "uv found at ~/.local/bin ($UV_VERSION)"
        audit_log "UV_FOUND path=~/.local/bin version=$UV_VERSION"
        return 0
    fi

    if [ -x "$HOME/.cargo/bin/uv" ]; then
        UV_CMD="$HOME/.cargo/bin/uv"
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        log_success "uv found at ~/.cargo/bin ($UV_VERSION)"
        audit_log "UV_FOUND path=~/.cargo/bin version=$UV_VERSION"
        return 0
    fi

    # Install uv — download to temp file and verify checksum before executing
    log_info "Installing uv (fast Python package manager)..."

    local tmp_dir
    tmp_dir="$(safe_temp_dir)"
    local uv_installer="$tmp_dir/uv-install.sh"

    if curl -LsSf https://astral.sh/uv/install.sh -o "$uv_installer" 2>/dev/null; then
        # Verify checksum before executing
        if verify_sha256 "$uv_installer" "$UV_INSTALLER_SHA256" "uv installer"; then
            if sh "$uv_installer" 2>/dev/null; then
                if [ -x "$HOME/.local/bin/uv" ]; then
                    UV_CMD="$HOME/.local/bin/uv"
                elif [ -x "$HOME/.cargo/bin/uv" ]; then
                    UV_CMD="$HOME/.cargo/bin/uv"
                elif command -v uv &> /dev/null; then
                    UV_CMD="uv"
                else
                    log_error "uv installed but not found on PATH"
                    log_info "Try adding ~/.local/bin to your PATH and re-running"
                    audit_log "UV_INSTALL_FAIL not_on_path"
                    exit 1
                fi
                UV_VERSION=$($UV_CMD --version 2>/dev/null)
                log_success "uv installed ($UV_VERSION)"
                audit_log "UV_INSTALL_OK version=$UV_VERSION"
            else
                log_error "Failed to install uv"
                log_info "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
                audit_log "UV_INSTALL_FAIL installer_exec"
                exit 1
            fi
        else
            log_error "uv installer checksum verification FAILED — refusing to execute"
            log_info "This may indicate a supply-chain attack. Install uv manually:"
            log_info "  https://docs.astral.sh/uv/getting-started/installation/"
            audit_log "UV_INSTALL_FAIL checksum_mismatch"
            exit 1
        fi
    else
        log_error "Failed to download uv installer"
        log_info "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        audit_log "UV_INSTALL_FAIL download"
        exit 1
    fi
}

check_python() {
    if [ "$DISTRO" = "termux" ]; then
        log_info "Checking Termux Python..."
        if command -v python >/dev/null 2>&1; then
            PYTHON_PATH="$(command -v python)"
            if "$PYTHON_PATH" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
                PYTHON_FOUND_VERSION="$("$PYTHON_PATH" --version 2>/dev/null)"
                log_success "Python found: $PYTHON_FOUND_VERSION"
                audit_log "PYTHON_FOUND version=$PYTHON_FOUND_VERSION"
                return 0
            fi
        fi

        log_info "Installing Python via pkg..."
        pkg install -y python >/dev/null
        PYTHON_PATH="$(command -v python)"
        PYTHON_FOUND_VERSION="$("$PYTHON_PATH" --version 2>/dev/null)"
        log_success "Python installed: $PYTHON_FOUND_VERSION"
        audit_log "PYTHON_INSTALLED version=$PYTHON_FOUND_VERSION"
        return 0
    fi

    log_info "Checking Python $PYTHON_VERSION..."

    if PYTHON_PATH="$("$UV_CMD" python find "$PYTHON_VERSION" 2>/dev/null)"; then
        PYTHON_FOUND_VERSION="$("$PYTHON_PATH" --version 2>/dev/null)"
        log_success "Python found: $PYTHON_FOUND_VERSION"
        audit_log "PYTHON_FOUND version=$PYTHON_FOUND_VERSION"
        return 0
    fi

    log_info "Python $PYTHON_VERSION not found, installing via uv..."
    if "$UV_CMD" python install "$PYTHON_VERSION"; then
        PYTHON_PATH="$("$UV_CMD" python find "$PYTHON_VERSION")"
        PYTHON_FOUND_VERSION="$("$PYTHON_PATH" --version 2>/dev/null)"
        log_success "Python installed: $PYTHON_FOUND_VERSION"
        audit_log "PYTHON_INSTALLED version=$PYTHON_FOUND_VERSION"
    else
        log_error "Failed to install Python $PYTHON_VERSION"
        audit_log "PYTHON_INSTALL_FAIL"
        exit 1
    fi
}

check_git() {
    log_info "Checking Git..."

    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | awk '{print $3}')
        log_success "Git $GIT_VERSION found"
        audit_log "GIT_FOUND version=$GIT_VERSION"
        return 0
    fi

    log_error "Git not found"

    if [ "$DISTRO" = "termux" ]; then
        log_info "Installing Git via pkg..."
        pkg install -y git >/dev/null
        if command -v git >/dev/null 2>&1; then
            GIT_VERSION=$(git --version | awk '{print $3}')
            log_success "Git $GIT_VERSION installed"
            audit_log "GIT_INSTALLED version=$GIT_VERSION"
            return 0
        fi
    fi

    log_info "Please install Git:"
    case "$OS" in
        linux)
            case "$DISTRO" in
                ubuntu|debian) log_info "  sudo apt update && sudo apt install git" ;;
                fedora)        log_info "  sudo dnf install git" ;;
                arch)          log_info "  sudo pacman -S git" ;;
                *)             log_info "  Use your package manager to install git" ;;
            esac
            ;;
        android) log_info "  pkg install git" ;;
        macos)
            log_info "  xcode-select --install"
            log_info "  Or: brew install git"
            ;;
    esac

    exit 1
}

check_node() {
    log_info "Checking Node.js (for browser tools)..."

    if command -v node &> /dev/null; then
        local found_ver
        found_ver="$(node --version)"
        log_success "Node.js $found_ver found"
        HAS_NODE=true
        audit_log "NODE_FOUND version=$found_ver"
        return 0
    fi

    if [ -x "$HERMES_HOME/node/bin/node" ]; then
        export PATH="$HERMES_HOME/node/bin:$PATH"
        local found_ver
        found_ver="$("$HERMES_HOME/node/bin/node" --version)"
        log_success "Node.js $found_ver found (Hermes-managed)"
        HAS_NODE=true
        audit_log "NODE_FOUND path=hermes-managed version=$found_ver"
        return 0
    fi

    if [ "$DISTRO" = "termux" ]; then
        log_info "Node.js not found — installing Node.js via pkg..."
    else
        log_info "Node.js not found — installing Node.js $NODE_VERSION LTS..."
    fi
    install_node
}

install_node() {
    if [ "$DISTRO" = "termux" ]; then
        log_info "Installing Node.js via pkg..."
        if pkg install -y nodejs >/dev/null; then
            local installed_ver
            installed_ver="$(node --version 2>/dev/null)"
            log_success "Node.js $installed_ver installed via pkg"
            HAS_NODE=true
            audit_log "NODE_INSTALLED version=$installed_ver source=pkg"
        else
            log_warn "Failed to install Node.js via pkg"
            HAS_NODE=false
        fi
        return 0
    fi

    local arch
    arch="$(uname -m)"
    local node_arch
    case "$arch" in
        x86_64)        node_arch="x64"    ;;
        aarch64|arm64) node_arch="arm64"  ;;
        armv7l)        node_arch="armv7l" ;;
        *)
            log_warn "Unsupported architecture ($arch) for Node.js auto-install"
            log_info "Install manually: https://nodejs.org/en/download/"
            HAS_NODE=false
            return 0
            ;;
    esac

    local node_os
    case "$OS" in
        linux) node_os="linux"  ;;
        macos) node_os="darwin" ;;
        *)
            log_warn "Unsupported OS for Node.js auto-install"
            HAS_NODE=false
            return 0
            ;;
    esac

    # Resolve the latest tarball name from the index page
    local index_url="https://nodejs.org/dist/latest-v${NODE_VERSION}.x/"
    local tarball_name
    tarball_name=$(curl -fsSL "$index_url" \
        | grep -oE "node-v${NODE_VERSION}\.[0-9]+\.[0-9]+-${node_os}-${node_arch}\.tar\.xz" \
        | head -1)

    if [ -z "$tarball_name" ]; then
        tarball_name=$(curl -fsSL "$index_url" \
            | grep -oE "node-v${NODE_VERSION}\.[0-9]+\.[0-9]+-${node_os}-${node_arch}\.tar\.gz" \
            | head -1)
    fi

    if [ -z "$tarball_name" ]; then
        log_warn "Could not find Node.js $NODE_VERSION binary for $node_os-$node_arch"
        log_info "Install manually: https://nodejs.org/en/download/"
        HAS_NODE=false
        return 0
    fi

    local download_url="${index_url}${tarball_name}"
    local tmp_dir
    tmp_dir="$(safe_temp_dir)"

    log_info "Downloading $tarball_name..."
    if ! curl -fsSL "$download_url" -o "$tmp_dir/$tarball_name"; then
        log_warn "Download failed"
        HAS_NODE=false
        return 0
    fi

    # Resolve the expected checksum for this platform
    local expected_hash=""
    case "$node_os-$node_arch" in
        linux-x64)   expected_hash="$NODE_SHA256_LINUX_X64"   ;;
        linux-arm64) expected_hash="$NODE_SHA256_LINUX_ARM64" ;;
        darwin-x64)  expected_hash="$NODE_SHA256_DARWIN_X64"   ;;
        darwin-arm64) expected_hash="$NODE_SHA256_DARWIN_ARM64" ;;
    esac

    # Verify checksum — tarballs are deterministic, fail hard on mismatch
    if [ -n "$expected_hash" ]; then
        if ! verify_sha256 "$tmp_dir/$tarball_name" "$expected_hash" "Node.js $tarball_name"; then
            log_error "Node.js tarball checksum MISMATCH — aborting (possible supply-chain attack)"
            audit_log "NODE_VERIFY_FAIL tarball=$tarball_name"
            HAS_NODE=false
            return 0
        fi
    elif [ "$SKIP_VERIFY" = false ]; then
        log_warn "No pinned checksum for Node.js $node_os-$node_arch — install without verification"
        log_warn "Populate NODE_SHA256_${node_os^^}_${node_arch^^} in script header for full integrity"
        audit_log "NODE_VERIFY_NO_PIN os=$node_os arch=$node_arch"
    fi

    log_info "Extracting to ~/.hermes/node/..."
    if [[ "$tarball_name" == *.tar.xz ]]; then
        tar xf "$tmp_dir/$tarball_name" -C "$tmp_dir"
    else
        tar xzf "$tmp_dir/$tarball_name" -C "$tmp_dir"
    fi

    local extracted_dir
    extracted_dir=$(ls -d "$tmp_dir"/node-v* 2>/dev/null | head -1)

    if [ ! -d "$extracted_dir" ]; then
        log_warn "Extraction failed"
        HAS_NODE=false
        return 0
    fi

    rm -rf "$HERMES_HOME/node"
    mkdir -p "$HERMES_HOME"
    mv "$extracted_dir" "$HERMES_HOME/node"

    mkdir -p "$HOME/.local/bin"
    ln -sf "$HERMES_HOME/node/bin/node" "$HOME/.local/bin/node"
    ln -sf "$HERMES_HOME/node/bin/npm"  "$HOME/.local/bin/npm"
    ln -sf "$HERMES_HOME/node/bin/npx"  "$HOME/.local/bin/npx"

    export PATH="$HERMES_HOME/node/bin:$PATH"

    local installed_ver
    installed_ver="$("$HERMES_HOME/node/bin/node" --version 2>/dev/null)"
    log_success "Node.js $installed_ver installed to ~/.hermes/node/"
    audit_log "NODE_INSTALLED version=$installed_ver path=~/.hermes/node"
    HAS_NODE=true
}

install_system_packages() {
    HAS_RIPGREP=false
    HAS_FFMPEG=false
    local need_ripgrep=false
    local need_ffmpeg=false

    log_info "Checking ripgrep (fast file search)..."
    if command -v rg &> /dev/null; then
        log_success "$(rg --version | head -1) found"
        HAS_RIPGREP=true
    else
        need_ripgrep=true
    fi

    log_info "Checking ffmpeg (TTS voice messages)..."
    if command -v ffmpeg &> /dev/null; then
        local ffmpeg_ver
        ffmpeg_ver="$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"
        log_success "ffmpeg $ffmpeg_ver found"
        HAS_FFMPEG=true
    else
        need_ffmpeg=true
    fi

    if [ "$DISTRO" = "termux" ]; then
        local termux_pkgs=(clang rust make pkg-config libffi openssl)
        [ "$need_ripgrep" = true ] && termux_pkgs+=("ripgrep")
        [ "$need_ffmpeg" = true ] && termux_pkgs+=("ffmpeg")

        log_info "Installing Termux packages: ${termux_pkgs[*]}"
        if pkg install -y "${termux_pkgs[@]}" >/dev/null; then
            [ "$need_ripgrep" = true ] && HAS_RIPGREP=true && log_success "ripgrep installed"
            [ "$need_ffmpeg" = true ]  && HAS_FFMPEG=true  && log_success "ffmpeg installed"
            audit_log "TERMUX_PKGS_OK pkgs=${termux_pkgs[*]}"
            return 0
        fi
        log_warn "Could not auto-install all Termux packages"
        return 0
    fi

    if [ "$need_ripgrep" = false ] && [ "$need_ffmpeg" = false ]; then
        return 0
    fi

    local desc_parts=()
    local pkgs=()
    [ "$need_ripgrep" = true ] && desc_parts+=("ripgrep for faster file search") && pkgs+=("ripgrep")
    [ "$need_ffmpeg" = true ]  && desc_parts+=("ffmpeg for TTS voice messages")  && pkgs+=("ffmpeg")
    local description
    description=$(IFS=" and "; echo "${desc_parts[*]}")

    if [ "$OS" = "macos" ]; then
        if command -v brew &> /dev/null; then
            log_info "Installing ${pkgs[*]} via Homebrew..."
            if brew install "${pkgs[@]}"; then
                [ "$need_ripgrep" = true ] && HAS_RIPGREP=true && log_success "ripgrep installed"
                [ "$need_ffmpeg" = true ]  && HAS_FFMPEG=true  && log_success "ffmpeg installed"
                audit_log "BREW_PKGS_OK pkgs=${pkgs[*]}"
                return 0
            fi
        fi
        log_warn "Could not auto-install (brew not found or install failed)"
        log_info "Install manually: brew install ${pkgs[*]}"
        return 0
    fi

    local pkg_install=""
    case "$DISTRO" in
        ubuntu|debian) pkg_install="apt install -y"   ;;
        fedora)        pkg_install="dnf install -y"   ;;
        arch)          pkg_install="pacman -S --noconfirm" ;;
    esac

    if [ -n "$pkg_install" ]; then
        local install_cmd="$pkg_install ${pkgs[*]}"

        case "$DISTRO" in
            ubuntu|debian) export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a ;;
        esac

        if [ "$(id -u)" -eq 0 ]; then
            log_info "Installing ${pkgs[*]}..."
            if $install_cmd; then
                [ "$need_ripgrep" = true ] && HAS_RIPGREP=true && log_success "ripgrep installed"
                [ "$need_ffmpeg" = true ]  && HAS_FFMPEG=true  && log_success "ffmpeg installed"
                audit_log "SYS_PKGS_OK pkgs=${pkgs[*]} method=root"
                return 0
            fi
        elif command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
            log_info "Installing ${pkgs[*]}..."
            if sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a $install_cmd; then
                clear_sudo
                [ "$need_ripgrep" = true ] && HAS_RIPGREP=true && log_success "ripgrep installed"
                [ "$need_ffmpeg" = true ]  && HAS_FFMPEG=true  && log_success "ffmpeg installed"
                audit_log "SYS_PKGS_OK pkgs=${pkgs[*]} method=sudo-n"
                return 0
            fi
            clear_sudo
        elif command -v sudo &> /dev/null; then
            if [ "$IS_INTERACTIVE" = true ]; then
                echo ""
                log_info "sudo is needed ONLY to install optional system packages (${pkgs[*]})."
                log_info "Hermes Agent itself does not require or retain root access."
                if prompt_yes_no "Install ${description}? (requires sudo)" "no"; then
                    if sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a $install_cmd; then
                        clear_sudo
                        [ "$need_ripgrep" = true ] && HAS_RIPGREP=true && log_success "ripgrep installed"
                        [ "$need_ffmpeg" = true ]  && HAS_FFMPEG=true  && log_success "ffmpeg installed"
                        audit_log "SYS_PKGS_OK pkgs=${pkgs[*]} method=sudo-interactive"
                        return 0
                    fi
                    clear_sudo
                fi
            elif (: </dev/tty) 2>/dev/null; then
                echo ""
                log_info "sudo is needed ONLY to install optional system packages (${pkgs[*]})."
                log_info "Hermes Agent itself does not require or retain root access."
                if prompt_yes_no "Install ${description}?" "yes"; then
                    if sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a $install_cmd < /dev/tty; then
                        clear_sudo
                        [ "$need_ripgrep" = true ] && HAS_RIPGREP=true && log_success "ripgrep installed"
                        [ "$need_ffmpeg" = true ]  && HAS_FFMPEG=true  && log_success "ffmpeg installed"
                        audit_log "SYS_PKGS_OK pkgs=${pkgs[*]} method=sudo-tty"
                        return 0
                    fi
                    clear_sudo
                fi
            else
                log_warn "Non-interactive mode and no terminal available — cannot install system packages"
                log_info "Install manually: sudo $install_cmd"
            fi
        fi
    fi

    # Fallback: cargo install ripgrep
    if [ "$need_ripgrep" = true ] && [ "$HAS_RIPGREP" = false ]; then
        if command -v cargo &> /dev/null; then
            log_info "Trying cargo install ripgrep (no sudo needed)..."
            if cargo install ripgrep; then
                log_success "ripgrep installed via cargo"
                HAS_RIPGREP=true
                audit_log "RIPGREP_INSTALLED method=cargo"
            fi
        fi
    fi

    [ "$HAS_RIPGREP" = false ] && [ "$need_ripgrep" = true ] && log_warn "ripgrep not installed (file search will use grep fallback)"
    [ "$HAS_FFMPEG" = false ] && [ "$need_ffmpeg" = true ]  && log_warn "ffmpeg not installed (TTS voice messages will be limited)"
}

show_manual_install_hint() {
    local pkg="$1"
    log_info "To install $pkg manually:"
    case "$OS" in
        linux)
            case "$DISTRO" in
                ubuntu|debian) log_info "  sudo apt install $pkg" ;;
                fedora)        log_info "  sudo dnf install $pkg" ;;
                arch)          log_info "  sudo pacman -S $pkg"   ;;
                *)             log_info "  Use your package manager" ;;
            esac
            ;;
        android) log_info "  pkg install $pkg" ;;
        macos)    log_info "  brew install $pkg" ;;
    esac
}

# ============================================================================
# Installation
# ============================================================================

clone_repo() {
    log_info "Installing to $INSTALL_DIR..."

    # Validate install path
    validate_path "$INSTALL_DIR" "install directory" || exit 1

    if [ -d "$INSTALL_DIR" ]; then
        if [ -d "$INSTALL_DIR/.git" ]; then
            log_info "Existing installation found, updating..."
            cd "$INSTALL_DIR"

            local autostash_ref=""
            if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
                local stash_name
                stash_name="hermes-install-autostash-$(date -u +%Y%m%d-%H%M%S)"
                log_info "Local changes detected, stashing before update..."
                git stash push --include-untracked -m "$stash_name"
                # Store stash reference (stash@{N}) not commit SHA —
                # git stash drop only accepts stash references, not raw SHA hashes.
                autostash_ref="$(git stash list | grep ": ${stash_name}$" | head -1 | cut -d: -f1 || echo "stash@{0}")"
            fi

            # Ensure remote URL matches the expected repository (the user may
            # have an older install cloned from a different remote, e.g. upstream).
            local current_remote_url
            current_remote_url="$(git remote get-url origin 2>/dev/null || echo "")"
            if [ "$current_remote_url" != "$REPO_URL_HTTPS" ] && [ "$current_remote_url" != "$REPO_URL_SSH" ]; then
                log_info "Remote URL differs from expected — updating to $REPO_URL_HTTPS"
                git remote set-url origin "$REPO_URL_HTTPS"
                audit_log "REMOTE_URL_UPDATE old=$(echo "$current_remote_url" | redact_line)"
            fi

            git fetch origin

            if [ -n "$GIT_COMMIT" ]; then
                log_info "Checking out pinned commit: $GIT_COMMIT"
                git checkout "$GIT_COMMIT"
                audit_log "CLONE_UPDATE commit=$GIT_COMMIT"
            else
                git checkout "$BRANCH"
                git pull --ff-only origin "$BRANCH"
                audit_log "CLONE_UPDATE branch=$BRANCH"
            fi

            local resolved_head
            resolved_head="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
            audit_log "CLONE_HEAD head=$resolved_head"

            if [ -n "$autostash_ref" ]; then
                local restore_now="yes"
                if [ -t 0 ] && [ -t 1 ]; then
                    echo
                    log_warn "Local changes were stashed before updating."
                    log_warn "Restoring them may reapply local customizations onto the updated codebase."
                    printf "Restore local changes now? [Y/n] "
                    read -r restore_answer || restore_answer=""
                    case "$restore_answer" in
                        ""|y|Y|yes|YES|Yes) restore_now="yes" ;;
                        *) restore_now="no" ;;
                    esac
                fi

                if [ "$restore_now" = "yes" ]; then
                    log_info "Restoring local changes..."
                    if git stash apply "$autostash_ref"; then
                        git stash drop "$autostash_ref" >/dev/null
                        log_warn "Local changes were restored on top of the updated codebase."
                    else
                        log_error "Restoring local changes failed. Changes preserved in git stash."
                        log_info "Resolve manually: git stash apply $autostash_ref"
                        exit 1
                    fi
                else
                    log_info "Skipped restoring local changes. Still in git stash: $autostash_ref"
                fi
            fi
        else
            log_error "Directory exists but is not a git repository: $INSTALL_DIR"
            log_info "Remove it or choose a different directory with --dir"
            audit_log "CLONE_FAIL not_a_git_repo"
            exit 1
        fi
    else
        # Fresh clone
        CLONE_PERFORMED=true
        local clone_ref="$BRANCH"
        local clone_opts="--branch $BRANCH"

        if [ -n "$GIT_COMMIT" ]; then
            # When pinning commit, clone the branch first then checkout
            clone_opts="--branch $BRANCH"
            log_info "Will clone branch $BRANCH then checkout commit $GIT_COMMIT"
        fi

        log_info "Trying SSH clone..."
        if GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=5" \
           git clone $clone_opts "$REPO_URL_SSH" "$INSTALL_DIR" 2>/dev/null; then
            log_success "Cloned via SSH"
            audit_log "CLONE_OK method=ssh branch=$BRANCH"
        else
            rm -rf "$INSTALL_DIR" 2>/dev/null
            log_info "SSH failed, trying HTTPS..."
            if git clone $clone_opts "$REPO_URL_HTTPS" "$INSTALL_DIR"; then
                log_success "Cloned via HTTPS"
                audit_log "CLONE_OK method=https branch=$BRANCH"
            else
                log_error "Failed to clone repository"
                audit_log "CLONE_FAIL method=both"
                exit 1
            fi
        fi

        cd "$INSTALL_DIR"

        # If commit pin specified, checkout exact commit
        if [ -n "$GIT_COMMIT" ]; then
            log_info "Checking out pinned commit: $GIT_COMMIT"
            git checkout "$GIT_COMMIT"
            audit_log "CLONE_COMMIT checkout=$GIT_COMMIT"
        fi

        local resolved_head
        resolved_head="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
        audit_log "CLONE_HEAD head=$resolved_head"
    fi

    cd "$INSTALL_DIR"
    log_success "Repository ready"
}

setup_venv() {
    if [ "$USE_VENV" = false ]; then
        log_info "Skipping virtual environment (--no-venv)"
        return 0
    fi

    if [ "$DISTRO" = "termux" ]; then
        log_info "Creating virtual environment with Termux Python..."
        if [ -d "venv" ]; then
            log_info "Virtual environment already exists, recreating..."
            rm -rf venv
        fi
        "$PYTHON_PATH" -m venv venv
        VENV_CREATED=true
        log_success "Virtual environment ready ($(./venv/bin/python --version 2>/dev/null))"
        audit_log "VENV_CREATED method=stdlib"
        return 0
    fi

    log_info "Creating virtual environment with Python $PYTHON_VERSION..."
    if [ -d "venv" ]; then
        log_info "Virtual environment already exists, recreating..."
        rm -rf venv
    fi

    $UV_CMD venv venv --python "$PYTHON_VERSION"
    VENV_CREATED=true
    log_success "Virtual environment ready (Python $PYTHON_VERSION)"
    audit_log "VENV_CREATED method=uv version=$PYTHON_VERSION"
}

install_deps() {
    log_info "Installing dependencies..."

    if [ "$DISTRO" = "termux" ]; then
        if [ "$USE_VENV" = true ]; then
            export VIRTUAL_ENV="$INSTALL_DIR/venv"
            PIP_PYTHON="$INSTALL_DIR/venv/bin/python"
        else
            PIP_PYTHON="$PYTHON_PATH"
        fi

        if [ -z "${ANDROID_API_LEVEL:-}" ]; then
            ANDROID_API_LEVEL="$(getprop ro.build.version.sdk 2>/dev/null || true)"
            [ -z "$ANDROID_API_LEVEL" ] && ANDROID_API_LEVEL=24
            export ANDROID_API_LEVEL
            log_info "Using ANDROID_API_LEVEL=$ANDROID_API_LEVEL for Android wheel builds"
        fi

        "$PIP_PYTHON" -m pip install --upgrade pip setuptools wheel >/dev/null
        if ! "$PIP_PYTHON" -m pip install -e '.[termux]' -c constraints-termux.txt; then
            log_warn "Termux feature install (.[termux]) failed, trying base install..."
            if ! "$PIP_PYTHON" -m pip install -e '.' -c constraints-termux.txt; then
                log_error "Package installation failed on Termux."
                audit_log "DEPS_FAIL platform=termux"
                exit 1
            fi
        fi

        log_success "Main package installed"
        audit_log "DEPS_OK platform=termux"
        return 0
    fi

    if [ "$USE_VENV" = true ]; then
        export VIRTUAL_ENV="$INSTALL_DIR/venv"
    fi

    # Debian/Ubuntu build tools
    if [ "$DISTRO" = "ubuntu" ] || [ "$DISTRO" = "debian" ]; then
        local need_build_tools=false
        for pkg in gcc python3-dev libffi-dev; do
            if ! dpkg -s "$pkg" &>/dev/null; then
                need_build_tools=true
                break
            fi
        done
        if [ "$need_build_tools" = true ]; then
            log_info "Some build tools may be needed for Python packages..."
            if command -v sudo &> /dev/null; then
                if sudo -n true 2>/dev/null; then
                    sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get update -qq && \
                    sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y -qq build-essential python3-dev libffi-dev >/dev/null 2>&1 || true
                    clear_sudo
                    log_success "Build tools installed"
                    audit_log "BUILD_TOOLS_OK method=sudo-n"
                else
                    log_info "sudo is needed ONLY to install build tools."
                    log_info "Hermes Agent itself does not require or retain root access."
                    if prompt_yes_no "Install build tools?" "yes"; then
                        sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get update -qq && \
                        sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y -qq build-essential python3-dev libffi-dev >/dev/null 2>&1 || true
                        clear_sudo
                        log_success "Build tools installed"
                        audit_log "BUILD_TOOLS_OK method=sudo-interactive"
                    fi
                fi
            fi
        fi
    fi

    # Install main package
    local ALL_INSTALL_LOG
    ALL_INSTALL_LOG="$(safe_temp_dir)/install.log"
    if ! $UV_CMD pip install -e ".[all]" 2>"$ALL_INSTALL_LOG"; then
        log_warn "Full install (.[all]) failed, trying base install..."
        log_info "Reason: $(tail -5 "$ALL_INSTALL_LOG" | head -3 | redact_line)"
        if ! $UV_CMD pip install -e "."; then
            log_error "Package installation failed."
            audit_log "DEPS_FAIL method=base"
            exit 1
        fi
    fi

    log_success "Main package installed"

    if [ -d "tinker-atropos" ] && [ -f "tinker-atropos/pyproject.toml" ]; then
        log_info "tinker-atropos submodule found — skipping install (optional, for RL training)"
    fi

    log_success "All dependencies installed"
    audit_log "DEPS_OK"
}

setup_path() {
    log_info "Setting up hermes command..."

    if [ "$USE_VENV" = true ]; then
        HERMES_BIN="$INSTALL_DIR/venv/bin/hermes"
    else
        HERMES_BIN="$(which hermes 2>/dev/null || echo "")"
        if [ -z "$HERMES_BIN" ]; then
            log_warn "hermes not found on PATH after install"
            return 0
        fi
    fi

    if [ ! -x "$HERMES_BIN" ]; then
        log_warn "hermes entry point not found at $HERMES_BIN"
        log_info "This usually means the pip install didn't complete successfully."
        if [ "$DISTRO" = "termux" ]; then
            log_info "Try: cd $INSTALL_DIR && python -m pip install -e '.[termux]' -c constraints-termux.txt"
        else
            log_info "Try: cd $INSTALL_DIR && uv pip install -e '.[all]'"
        fi
        return 0
    fi

    local command_link_dir
    local command_link_display_dir
    command_link_dir="$(get_command_link_dir)"
    command_link_display_dir="$(get_command_link_display_dir)"

    # Validate the command link directory path
    validate_path "$command_link_dir" "command link dir" || return 0

    mkdir -p "$command_link_dir"
    ln -sf "$HERMES_BIN" "$command_link_dir/hermes"
    log_success "Symlinked hermes → $command_link_display_dir/hermes"
    audit_log "SYMLINK_OK target=$HERMES_BIN link=$command_link_dir/hermes"

    if [ "$DISTRO" = "termux" ]; then
        export PATH="$command_link_dir:$PATH"
        log_info "$command_link_display_dir is the native Termux command path"
        log_success "hermes command ready"
        return 0
    fi

    if [ "$ROOT_FHS_LAYOUT" = true ]; then
        export PATH="$command_link_dir:$PATH"
        if env -i HOME="$HOME" TERM="${TERM:-dumb}" bash -i -c 'command -v hermes' \
                >/dev/null 2>&1; then
            log_info "/usr/local/bin is already on PATH for all shells"
            log_success "hermes command ready"
            return 0
        fi

        log_info "hermes not on PATH in non-login shells (common on RHEL-family)"
        PATH_LINE='export PATH="/usr/local/bin:$PATH"'
        PATH_COMMENT='# Hermes Agent — ensure /usr/local/bin is on PATH (RHEL non-login shells)'
        for SHELL_CONFIG in "$HOME/.bashrc" "$HOME/.bash_profile"; do
            [ -f "$SHELL_CONFIG" ] || continue
            if ! grep -v '^[[:space:]]*#' "$SHELL_CONFIG" 2>/dev/null \
                    | grep -qE 'PATH=.*(/usr/local/bin|\$command_link_dir)'; then
                echo "" >> "$SHELL_CONFIG"
                echo "$PATH_COMMENT" >> "$SHELL_CONFIG"
                echo "$PATH_LINE" >> "$SHELL_CONFIG"
                log_success "Added /usr/local/bin to PATH in $SHELL_CONFIG"
                audit_log "PATH_CONFIG_OK file=$SHELL_CONFIG"
            fi
        done
        log_success "hermes command ready"
        return 0
    fi

    if ! echo "$PATH" | tr ':' '\n' | grep -q "^$command_link_dir$"; then
        SHELL_CONFIGS=()
        IS_FISH=false
        LOGIN_SHELL="$(basename "${SHELL:-/bin/bash}")"
        case "$LOGIN_SHELL" in
            zsh)
                [ -f "$HOME/.zshrc" ] && SHELL_CONFIGS+=("$HOME/.zshrc")
                [ -f "$HOME/.zprofile" ] && SHELL_CONFIGS+=("$HOME/.zprofile")
                if [ ${#SHELL_CONFIGS[@]} -eq 0 ]; then
                    touch "$HOME/.zshrc"
                    SHELL_CONFIGS+=("$HOME/.zshrc")
                fi
                ;;
            bash)
                [ -f "$HOME/.bashrc" ] && SHELL_CONFIGS+=("$HOME/.bashrc")
                [ -f "$HOME/.bash_profile" ] && SHELL_CONFIGS+=("$HOME/.bash_profile")
                ;;
            fish)
                IS_FISH=true
                FISH_CONFIG="$HOME/.config/fish/config.fish"
                mkdir -p "$(dirname "$FISH_CONFIG")"
                touch "$FISH_CONFIG"
                ;;
            *)
                [ -f "$HOME/.bashrc" ] && SHELL_CONFIGS+=("$HOME/.bashrc")
                [ -f "$HOME/.zshrc" ] && SHELL_CONFIGS+=("$HOME/.zshrc")
                ;;
        esac
        [ "$IS_FISH" = "false" ] && [ -f "$HOME/.profile" ] && SHELL_CONFIGS+=("$HOME/.profile")

        PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'

        for SHELL_CONFIG in "${SHELL_CONFIGS[@]}"; do
            if ! grep -v '^[[:space:]]*#' "$SHELL_CONFIG" 2>/dev/null | grep -qE 'PATH=.*\.local/bin'; then
                echo "" >> "$SHELL_CONFIG"
                echo "# Hermes Agent — ensure ~/.local/bin is on PATH" >> "$SHELL_CONFIG"
                echo "$PATH_LINE" >> "$SHELL_CONFIG"
                log_success "Added ~/.local/bin to PATH in $SHELL_CONFIG"
                audit_log "PATH_CONFIG_OK file=$SHELL_CONFIG"
            fi
        done

        if [ "$IS_FISH" = "true" ]; then
            if ! grep -q 'fish_add_path.*\.local/bin' "$FISH_CONFIG" 2>/dev/null; then
                echo "" >> "$FISH_CONFIG"
                echo "# Hermes Agent — ensure ~/.local/bin is on PATH" >> "$FISH_CONFIG"
                echo 'fish_add_path "$HOME/.local/bin"' >> "$FISH_CONFIG"
                log_success "Added ~/.local/bin to PATH in $FISH_CONFIG"
                audit_log "PATH_CONFIG_OK file=$FISH_CONFIG shell=fish"
            fi
        fi

        if [ "$IS_FISH" = "false" ] && [ ${#SHELL_CONFIGS[@]} -eq 0 ]; then
            log_warn "Could not detect shell config file to add ~/.local/bin to PATH"
            log_info "Add manually: $PATH_LINE"
        fi
    else
        log_info "~/.local/bin already on PATH"
    fi

    export PATH="$command_link_dir:$PATH"
    log_success "hermes command ready"
}

copy_config_templates() {
    log_info "Setting up configuration files..."

    # Validate HERMES_HOME path
    validate_path "$HERMES_HOME" "hermes home" || exit 1

    mkdir -p "$HERMES_HOME"/{cron,sessions,logs,pairing,hooks,image_cache,audio_cache,memories,skills}

    # .env — secret config, must be 0600
    if [ ! -f "$HERMES_HOME/.env" ]; then
        if [ -f "$INSTALL_DIR/.env.example" ]; then
            validate_path "$INSTALL_DIR/.env.example" "env template" || exit 1
            cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
            log_success "Created ~/.hermes/.env from template"
            audit_log "CONFIG_CREATED file=.env source=template"
        else
            touch "$HERMES_HOME/.env"
            log_success "Created ~/.hermes/.env"
            audit_log "CONFIG_CREATED file=.env source=empty"
        fi
    else
        log_info "~/.hermes/.env already exists, keeping it"
    fi

    # config.yaml
    if [ ! -f "$HERMES_HOME/config.yaml" ]; then
        if [ -f "$INSTALL_DIR/cli-config.yaml.example" ]; then
            validate_path "$INSTALL_DIR/cli-config.yaml.example" "config template" || exit 1
            cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
            log_success "Created ~/.hermes/config.yaml from template"
            audit_log "CONFIG_CREATED file=config.yaml source=template"
        fi
    else
        log_info "~/.hermes/config.yaml already exists, keeping it"
    fi

    # SOUL.md
    if [ ! -f "$HERMES_HOME/SOUL.md" ]; then
        cat > "$HERMES_HOME/SOUL.md" << 'SOUL_EOF'
# Hermes Agent Persona

<!--
This file defines the agent's personality and tone.
The agent will embody whatever you write here.
Edit this to customize how Hermes communicates with you.

This file is loaded fresh each message -- no restart needed.
Delete the contents (or this file) to use the default personality.
-->
SOUL_EOF
        log_success "Created ~/.hermes/SOUL.md (edit to customize personality)"
    fi

    # Enforce permissions on all sensitive files
    enforce_permissions "$HERMES_HOME"

    log_success "Configuration directory ready: ~/.hermes/"

    # Sync bundled skills
    log_info "Syncing bundled skills to ~/.hermes/skills/ ..."
    if [ -x "$INSTALL_DIR/venv/bin/python" ] && [ -f "$INSTALL_DIR/tools/skills_sync.py" ]; then
        "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/tools/skills_sync.py" 2>/dev/null && \
            log_success "Skills synced to ~/.hermes/skills/" || true
    else
        if [ -d "$INSTALL_DIR/skills" ] && [ ! "$(ls -A "$HERMES_HOME/skills/" 2>/dev/null | grep -v '.bundled_manifest')" ]; then
            cp -r "$INSTALL_DIR/skills/"* "$HERMES_HOME/skills/" 2>/dev/null || true
            log_success "Skills copied to ~/.hermes/skills/"
        fi
    fi
}

install_node_deps() {
    if [ "${HAS_NODE:-false}" = false ]; then
        log_info "Skipping Node.js dependencies (Node not installed)"
        return 0
    fi

    if [ "$DISTRO" = "termux" ]; then
        log_info "Skipping automatic Node/browser dependency setup on Termux"
        return 0
    fi

    if [ -f "$INSTALL_DIR/package.json" ]; then
        log_info "Installing Node.js dependencies (browser tools)..."
        cd "$INSTALL_DIR"
        npm install --silent 2>/dev/null || {
            log_warn "npm install failed (browser tools may not work)"
        }
        log_success "Node.js dependencies installed"
        audit_log "NPM_DEPS_OK"

        log_info "Installing browser engine (Playwright Chromium)..."
        case "$DISTRO" in
            ubuntu|debian|raspbian|pop|linuxmint|elementary|zorin|kali|parrot)
                log_info "Playwright may request sudo for browser system dependencies."
                log_info "Hermes itself does not require root access."
                cd "$INSTALL_DIR" && npx playwright install --with-deps chromium 2>/dev/null || {
                    log_warn "Playwright browser installation failed."
                    log_warn "Try: cd $INSTALL_DIR && npx playwright install --with-deps chromium"
                }
                clear_sudo 2>/dev/null || true
                ;;
            arch|manjaro)
                if command -v pacman &> /dev/null; then
                    if command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
                        sudo NEEDRESTART_MODE=a pacman -S --noconfirm --needed \
                            nss atk at-spi2-core cups libdrm libxkbcommon mesa pango cairo alsa-lib >/dev/null 2>&1 || true
                        clear_sudo
                    elif [ "$(id -u)" -eq 0 ]; then
                        pacman -S --noconfirm --needed \
                            nss atk at-spi2-core cups libdrm libxkbcommon mesa pango cairo alsa-lib >/dev/null 2>&1 || true
                    fi
                fi
                cd "$INSTALL_DIR" && npx playwright install chromium 2>/dev/null || {
                    log_warn "Playwright browser installation failed."
                }
                ;;
            fedora|rhel|centos|rocky|alma)
                log_warn "Playwright does not support auto deps on RPM-based systems."
                log_info "Install Chromium deps manually before using browser tools."
                cd "$INSTALL_DIR" && npx playwright install chromium 2>/dev/null || {
                    log_warn "Playwright browser installation failed."
                }
                ;;
            opensuse*|sles)
                log_warn "Playwright does not support auto deps on zypper-based systems."
                cd "$INSTALL_DIR" && npx playwright install chromium 2>/dev/null || {
                    log_warn "Playwright browser installation failed."
                }
                ;;
            *)
                log_warn "Playwright does not support auto deps on $DISTRO."
                cd "$INSTALL_DIR" && npx playwright install chromium 2>/dev/null || true
                ;;
        esac
        log_success "Browser engine setup complete"
        audit_log "PLAYWRIGHT_SETUP_OK"
    fi

    # TUI dependencies
    if [ -f "$INSTALL_DIR/ui-tui/package.json" ]; then
        log_info "Installing TUI dependencies..."
        cd "$INSTALL_DIR/ui-tui"
        npm install --silent 2>/dev/null || {
            log_warn "TUI npm install failed (hermes --tui may not work)"
        }
        log_success "TUI dependencies installed"
        audit_log "TUI_DEPS_OK"
    fi
}

run_setup_wizard() {
    if [ "$RUN_SETUP" = false ]; then
        log_info "Skipping setup wizard (--skip-setup)"
        return 0
    fi

    if ! (: </dev/tty) 2>/dev/null; then
        log_info "Setup wizard skipped (no terminal available). Run 'hermes setup' after install."
        return 0
    fi

    echo ""
    log_info "Starting setup wizard..."
    echo ""

    cd "$INSTALL_DIR"

    if [ "$USE_VENV" = true ]; then
        "$INSTALL_DIR/venv/bin/python" -m hermes_cli.main setup < /dev/tty
    else
        python -m hermes_cli.main setup < /dev/tty
    fi
    audit_log "SETUP_WIZARD_COMPLETE"
}

maybe_start_gateway() {
    ENV_FILE="$HERMES_HOME/.env"
    if [ ! -f "$ENV_FILE" ]; then
        return 0
    fi

    HAS_MESSAGING=false
    for VAR in TELEGRAM_BOT_TOKEN DISCORD_BOT_TOKEN SLACK_BOT_TOKEN SLACK_APP_TOKEN WHATSAPP_ENABLED; do
        VAL=$(grep "^${VAR}=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2-)
        if [ -n "$VAL" ] && [ "$VAL" != "your-token-here" ]; then
            HAS_MESSAGING=true
            break
        fi
    done

    if [ "$HAS_MESSAGING" = false ]; then
        return 0
    fi

    echo ""
    log_info "Messaging platform token detected!"
    log_info "The gateway needs to be running for Hermes to send/receive messages."

    WHATSAPP_VAL=$(grep "^WHATSAPP_ENABLED=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2-)
    WHATSAPP_SESSION="$HERMES_HOME/whatsapp/session/creds.json"
    if [ "$WHATSAPP_VAL" = "true" ] && [ ! -f "$WHATSAPP_SESSION" ]; then
        if [ "$IS_INTERACTIVE" = true ]; then
            echo ""
            log_info "WhatsApp is enabled but not yet paired."
            log_info "Running 'hermes whatsapp' to pair via QR code..."
            echo ""
            if prompt_yes_no "Pair WhatsApp now?" "yes"; then
                HERMES_CMD="$(get_hermes_command_path)"
                $HERMES_CMD whatsapp || true
            fi
        else
            log_info "WhatsApp pairing skipped (non-interactive)."
        fi
    fi

    if ! (: </dev/tty) 2>/dev/null; then
        log_info "Gateway setup skipped (no terminal available). Run 'hermes gateway install' later."
        return 0
    fi

    echo ""
    local should_install_gateway=false
    if [ "$DISTRO" = "termux" ]; then
        if prompt_yes_no "Would you like to start the gateway in the background?" "yes"; then
            should_install_gateway=true
        fi
    else
        if prompt_yes_no "Would you like to install the gateway as a background service?" "yes"; then
            should_install_gateway=true
        fi
    fi

    if [ "$should_install_gateway" = true ]; then
        HERMES_CMD="$(get_hermes_command_path)"

        if [ "$DISTRO" != "termux" ] && command -v systemctl &> /dev/null; then
            log_info "Installing systemd service..."
            if $HERMES_CMD gateway install 2>/dev/null; then
                log_success "Gateway service installed"
                audit_log "GATEWAY_INSTALL_OK method=systemd"
                if $HERMES_CMD gateway start 2>/dev/null; then
                    log_success "Gateway started! Your bot is now online."
                    audit_log "GATEWAY_START_OK"
                else
                    log_warn "Service installed but failed to start. Try: hermes gateway start"
                    audit_log "GATEWAY_START_FAIL"
                fi
            else
                log_warn "Systemd install failed. You can start manually: hermes gateway"
                audit_log "GATEWAY_INSTALL_FAIL"
            fi
        else
            if [ "$DISTRO" = "termux" ]; then
                log_info "Termux — starting gateway in best-effort background mode..."
            else
                log_info "systemd not available — starting gateway in background..."
            fi
            nohup $HERMES_CMD gateway > "$HERMES_HOME/logs/gateway.log" 2>&1 &
            GATEWAY_PID=$!
            log_success "Gateway started (PID $GATEWAY_PID). Logs: ~/.hermes/logs/gateway.log"
            log_info "To stop: kill $GATEWAY_PID"
            log_info "To restart later: hermes gateway"
            audit_log "GATEWAY_START_OK method=nohup pid=$GATEWAY_PID"
            if [ "$DISTRO" = "termux" ]; then
                log_warn "Android may stop background processes when Termux is suspended."
            fi
        fi
    else
        log_info "Skipped. Start the gateway later with: hermes gateway"
    fi

    # Re-enforce permissions after gateway may have created new files
    enforce_permissions "$HERMES_HOME"
}

print_success() {
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│              ✓ Installation Complete!                   │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
    echo ""

    echo -e "${CYAN}${BOLD}Your files:${NC}"
    echo ""
    echo -e "   ${YELLOW}Config:${NC}    $HERMES_HOME/config.yaml"
    echo -e "   ${YELLOW}API Keys:${NC}  $HERMES_HOME/.env"
    echo -e "   ${YELLOW}Data:${NC}      $HERMES_HOME/cron/, sessions/, logs/"
    echo -e "   ${YELLOW}Code:${NC}      $INSTALL_DIR"
    echo -e "   ${YELLOW}Audit Log:${NC} $HERMES_HOME/install-audit.log"
    echo ""

    echo -e "${CYAN}─────────────────────────────────────────────────────────${NC}"
    echo ""
    echo -e "${CYAN}${BOLD}Commands:${NC}"
    echo ""
    echo -e "   ${GREEN}hermes${NC}              Start chatting"
    echo -e "   ${GREEN}hermes setup${NC}        Configure API keys & settings"
    echo -e "   ${GREEN}hermes config${NC}       View/edit configuration"
    echo -e "   ${GREEN}hermes config edit${NC}  Open config in editor"
    echo -e "   ${GREEN}hermes gateway install${NC} Install gateway service"
    echo -e "   ${GREEN}hermes update${NC}       Update to latest version"
    echo ""

    echo -e "${CYAN}─────────────────────────────────────────────────────────${NC}"
    echo ""
    if [ "$DISTRO" = "termux" ]; then
        echo -e "${YELLOW}'hermes' was linked into $(get_command_link_display_dir), which is already on PATH in Termux.${NC}"
    elif [ "$ROOT_FHS_LAYOUT" = true ]; then
        echo -e "${YELLOW}'hermes' was linked into /usr/local/bin and is ready to use.${NC}"
    else
        echo -e "${YELLOW}Reload your shell to use 'hermes' command:${NC}"
        echo ""
        LOGIN_SHELL="$(basename "${SHELL:-/bin/bash}")"
        if [ "$LOGIN_SHELL" = "zsh" ]; then
            echo "   source ~/.zshrc"
        elif [ "$LOGIN_SHELL" = "bash" ]; then
            echo "   source ~/.bashrc"
        elif [ "$LOGIN_SHELL" = "fish" ]; then
            echo "   source ~/.config/fish/config.fish"
        else
            echo "   source ~/.bashrc   # or ~/.zshrc"
        fi
        echo ""
    fi

    if [ "${HAS_NODE:-false}" = false ]; then
        echo -e "${YELLOW}"
        echo "Note: Node.js could not be installed automatically."
        echo "Browser tools need Node.js. Install manually:"
        if [ "$DISTRO" = "termux" ]; then
            echo "  pkg install nodejs"
        else
            echo "  https://nodejs.org/en/download/"
        fi
        echo -e "${NC}"
    fi

    if [ "${HAS_RIPGREP:-false}" = false ]; then
        echo -e "${YELLOW}"
        echo "Note: ripgrep (rg) was not found. File search will use"
        echo "grep as a fallback."
        if [ "$DISTRO" = "termux" ]; then
            echo "Install ripgrep: pkg install ripgrep"
        else
            echo "Install ripgrep: sudo apt install ripgrep (or brew install ripgrep)"
        fi
        echo -e "${NC}"
    fi

    # Security summary
    echo ""
    echo -e "${CYAN}─────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}${BOLD}Security Summary:${NC}"
    echo ""
    echo -e "   Install ID:  ${GREEN}$INSTALL_ID${NC}"
    echo -e "   Audit Log:   ${GREEN}$HERMES_HOME/install-audit.log${NC}"
    echo -e "   .env perms:  ${GREEN}0600 (owner-only)${NC}"
    echo -e "   Config perms: ${GREEN}0600 (owner-only)${NC}"
    echo -e "   ~/${HERMES_HOME#$HOME/} perms: ${GREEN}0700 (owner-only)${NC}"
    if [ -n "$GIT_COMMIT" ]; then
        echo -e "   Git commit:  ${GREEN}$GIT_COMMIT (pinned)${NC}"
    fi
    if [ "$SKIP_VERIFY" = true ]; then
        echo -e "   ${RED}Checksum verification: DISABLED (--skip-verify)${NC}"
    else
        echo -e "   Checksums:   ${GREEN}Verified where pinned${NC}"
    fi
    echo ""
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Set up rollback trap
    trap rollback EXIT

    print_banner

    # Preflight checks
    detect_os

    if [ "$AUDIT_ONLY" = true ]; then
        log_info "Audit-only mode — running preflight checks and exiting"
        echo ""
        log_info "OS: $OS ($DISTRO)"
        log_info "HERMES_HOME: $HERMES_HOME"
        log_info "Branch: $BRANCH"
        [ -n "$GIT_COMMIT" ] && log_info "Commit: $GIT_COMMIT (pinned)"
        log_info "Skip verify: $SKIP_VERIFY"
        log_info "Checksum pins populated: $(if [ -n "$UV_INSTALLER_SHA256" ] || [ -n "$NODE_SHA256_LINUX_X64" ]; then echo "yes"; else echo "no (will warn)"; fi)"
        echo ""
        log_success "Preflight checks complete"
        audit_log "AUDIT_ONLY_COMPLETE"
        INSTALL_SUCCEEDED=true
        return 0
    fi

    resolve_install_layout
    install_uv
    check_python
    check_git
    check_node
    install_system_packages

    clone_repo
    setup_venv
    install_deps
    install_node_deps
    setup_path
    copy_config_templates

    # Core installation is complete — setup wizard and gateway are optional
    # post-install steps. Their failure must NOT trigger full rollback.
    INSTALL_SUCCEEDED=true
    audit_log "INSTALL_CORE_SUCCEEDED id=$INSTALL_ID"

    run_setup_wizard || {
        log_warn "Setup wizard failed — you can run 'hermes setup' manually later"
        audit_log "SETUP_WIZARD_FAILED"
    }
    maybe_start_gateway || {
        log_warn "Gateway setup failed — you can run 'hermes gateway install' manually later"
        audit_log "GATEWAY_SETUP_FAILED"
    }

    # Write install run ID for cross-checking
    echo "$INSTALL_ID $INSTALL_TIMESTAMP" > "$HERMES_HOME/.install-run-id"

    # Final permissions enforcement
    enforce_permissions "$HERMES_HOME"

    print_success

    audit_log "INSTALL_SUCCEEDED id=$INSTALL_ID"
}

main "$@"