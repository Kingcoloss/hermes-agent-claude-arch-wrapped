#!/bin/bash
# ============================================================================
# Hermes Agent Uninstaller
# ============================================================================
# Security-hardened uninstall script for Hermes Agent.
# Selectively removes installation artifacts with interactive confirmation,
# audit logging, and safe path validation.
#
# Usage:
#   bash uninstall.sh              # Remove code + command only (preserves data)
#   bash uninstall.sh --all        # Remove everything
#   bash uninstall.sh --yes --all  # Non-interactive full removal
#   bash uninstall.sh --config     # Remove ~/.hermes/ data only
#
# ============================================================================

set -euo pipefail

# ============================================================================
# Shell hardening
# ============================================================================

umask 0077

# Generate unique uninstall run ID
UNINSTALL_ID="$(uuidgen 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -d '-' || head -c32 /dev/urandom 2>/dev/null | xxd -p -c32 2>/dev/null || printf '%s%s' "$$" "$(date +%s)")"
UNINSTALL_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date +%s)"

# ============================================================================
# Colors
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;34m'
NC='\033[0m'
BOLD='\033[1m'

# ============================================================================
# Configuration
# ============================================================================

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
REMOVE_CODE=false
REMOVE_CONFIG=false
REMOVE_NODE=false
REMOVE_COMMAND=false
REMOVE_GATEWAY=false
AUTO_YES=false

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
        --code)
            REMOVE_CODE=true
            shift
            ;;
        --config)
            REMOVE_CONFIG=true
            shift
            ;;
        --node)
            REMOVE_NODE=true
            shift
            ;;
        --command)
            REMOVE_COMMAND=true
            shift
            ;;
        --gateway)
            REMOVE_GATEWAY=true
            shift
            ;;
        --all)
            REMOVE_CODE=true
            REMOVE_CONFIG=true
            REMOVE_NODE=true
            REMOVE_COMMAND=true
            REMOVE_GATEWAY=true
            shift
            ;;
        --yes|-y)
            AUTO_YES=true
            shift
            ;;
        --hermes-home)
            HERMES_HOME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Hermes Agent Uninstaller"
            echo ""
            echo "Usage: uninstall.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --code           Remove installed repo code + venv"
            echo "  --config         Remove ~/.hermes/ data directory (config, sessions, logs)"
            echo "  --node           Remove Hermes-managed Node.js at ~/.hermes/node/"
            echo "  --command        Remove hermes command symlink from PATH"
            echo "  --gateway        Stop and uninstall gateway service"
            echo "  --all            Remove everything (equivalent to all flags above)"
            echo "  --yes, -y       Skip confirmation prompts"
            echo "  --hermes-home    Data directory (default: ~/.hermes)"
            echo "  -h, --help       Show this help"
            echo ""
            echo "Default (no flags): remove code + command symlink only."
            echo "This preserves user data (sessions, config, memories, skills)."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default: remove code + command only (preserve user data)
if [ "$REMOVE_CODE" = false ] && [ "$REMOVE_CONFIG" = false ] && \
   [ "$REMOVE_NODE" = false ] && [ "$REMOVE_COMMAND" = false ] && \
   [ "$REMOVE_GATEWAY" = false ]; then
    REMOVE_CODE=true
    REMOVE_COMMAND=true
fi

# ============================================================================
# Security helper functions
# ============================================================================

audit_log() {
    local audit_file="$HERMES_HOME/install-audit.log"
    if [ -d "$(dirname "$audit_file")" ]; then
        local msg="[$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date +%s)] [uninstall-${UNINSTALL_ID}] $*"
        echo "$msg" >> "$audit_file" 2>/dev/null || true
    fi
}

redact_line() {
    sed -E \
        -e 's/sk-[A-Za-z0-9_-]{10,}/REDACTED_sk/g' \
        -e 's/ghp_[A-Za-z0-9]{10,}/REDACTED_ghp/g' \
        -e 's/AKIA[A-Z0-9]{16}/REDACTED_AKIA/g' \
        -e 's/xox[baprs]-[A-Za-z0-9-]{10,}/REDACTED_xox/g' \
        -e 's/([A-Z0-9_]{0,50}(API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)[A-Z0-9_]{0,50})=[^[:space:]]+/\1=REDACTED/g' \
        -e 's/Bearer [A-Za-z0-9._-]+/Bearer REDACTED/g'
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

validate_path() {
    local path_str="$1"
    case "$path_str" in
        *../* | */..* | *".. "* | *" .."*)
            log_error "Path traversal rejected: $path_str"
            audit_log "PATH_REJECTED traversal path=$(echo "$path_str" | redact_line)"
            return 1
            ;;
    esac
    return 0
}

prompt_yes_no() {
    local question="$1"
    local default="${2:-no}"
    local prompt_suffix
    local answer=""

    if [ "$AUTO_YES" = true ]; then
        case "$default" in
            [yY]|[yY][eE][sS]) return 0 ;;
            *) return 1 ;;
        esac
    fi

    case "$default" in
        [yY]|[yY][eE][sS]) prompt_suffix="[Y/n]" ;;
        *) prompt_suffix="[y/N]" ;;
    esac

    if [ "$IS_INTERACTIVE" = true ]; then
        read -r -p "$question $prompt_suffix " answer || answer=""
    elif [ -r /dev/tty ] && [ -w /dev/tty ]; then
        printf "%s %s " "$question" "$prompt_suffix" > /dev/tty
        IFS= read -r answer < /dev/tty || answer=""
    else
        # Non-interactive, no tty — use default
        case "$default" in
            [yY]|[yY][eE][sS]) return 0 ;;
            *) return 1 ;;
        esac
    fi

    answer="${answer#"${answer%%[![:space:]]*}"}"
    answer="${answer%"${answer##*[![:space:]]}"}"

    if [ -z "$answer" ]; then
        case "$default" in
            [yY]|[yY][eE][sS]) return 0 ;;
            *) return 1 ;;
        esac
    fi

    case "$answer" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

# ============================================================================
# Detect installation layout
# ============================================================================

detect_install_layout() {
    # Find INSTALL_DIR — check common locations
    INSTALL_DIR=""

    # Check HERMES_HOME/hermes-agent first (default layout)
    if [ -d "$HERMES_HOME/hermes-agent" ]; then
        INSTALL_DIR="$HERMES_HOME/hermes-agent"
    fi

    # Check FHS layout
    if [ -d "/usr/local/lib/hermes-agent" ]; then
        if [ -z "$INSTALL_DIR" ] || [ "$(id -u)" -eq 0 ]; then
            INSTALL_DIR="/usr/local/lib/hermes-agent"
        fi
    fi

    # Determine command link location
    COMMAND_LINK_DIR=""
    if [ -L "$HOME/.local/bin/hermes" ]; then
        COMMAND_LINK_DIR="$HOME/.local/bin"
    elif [ -L "/usr/local/bin/hermes" ]; then
        COMMAND_LINK_DIR="/usr/local/bin"
    elif [ -L "${PREFIX:-/usr}/bin/hermes" ]; then
        COMMAND_LINK_DIR="${PREFIX:-/usr}/bin"
    fi
}

# ============================================================================
# Removal functions
# ============================================================================

stop_gateway() {
    log_info "Checking for running gateway..."

    # Check systemd service first
    if command -v systemctl &>/dev/null; then
        if systemctl is-active --quiet hermes-gateway 2>/dev/null; then
            log_info "Stopping hermes-gateway systemd service..."
            sudo systemctl stop hermes-gateway 2>/dev/null || true
            sudo systemctl disable hermes-gateway 2>/dev/null || true
            sudo -k 2>/dev/null || true
            log_success "Gateway systemd service stopped and disabled"
            audit_log "GATEWAY_STOPPED method=systemd"
            return 0
        fi
    fi

    # Check for background process
    local gw_pid
    gw_pid="$(pgrep -f "hermes gateway" 2>/dev/null | head -1 || true)"
    if [ -n "$gw_pid" ]; then
        log_info "Stopping gateway process (PID $gw_pid)..."
        kill "$gw_pid" 2>/dev/null || true
        # Wait briefly for clean shutdown
        local tries=0
        while [ $tries -lt 10 ] && kill -0 "$gw_pid" 2>/dev/null; do
            sleep 1
            tries=$((tries + 1))
        done
        # Force kill if still running
        if kill -0 "$gw_pid" 2>/dev/null; then
            kill -9 "$gw_pid" 2>/dev/null || true
            log_warn "Force-killed gateway process"
        fi
        log_success "Gateway process stopped"
        audit_log "GATEWAY_STOPPED method=kill pid=$gw_pid"
    else
        log_info "No running gateway found"
    fi
}

remove_command_symlink() {
    if [ -z "$COMMAND_LINK_DIR" ]; then
        log_info "No hermes command symlink found"
        return 0
    fi

    local link_path="$COMMAND_LINK_DIR/hermes"
    if [ -L "$link_path" ]; then
        validate_path "$link_path" "command symlink" || return 0
        rm -f "$link_path"
        log_success "Removed command symlink: $link_path"
        audit_log "REMOVED symlink=$link_path"
    else
        log_info "Command symlink not found at $link_path"
    fi
}

remove_code() {
    if [ -z "$INSTALL_DIR" ]; then
        log_info "No installation directory found"
        return 0
    fi

    validate_path "$INSTALL_DIR" "install directory" || return 0

    if [ ! -d "$INSTALL_DIR" ]; then
        log_info "Install directory not found: $INSTALL_DIR"
        return 0
    fi

    # Record in manifest before removing
    record_manifest "code" "$INSTALL_DIR"

    rm -rf "$INSTALL_DIR"
    log_success "Removed code directory: $INSTALL_DIR"
    audit_log "REMOVED code_dir=$INSTALL_DIR"
}

remove_node() {
    local node_dir="$HERMES_HOME/node"
    if [ ! -d "$node_dir" ]; then
        log_info "No Hermes-managed Node.js installation found"
        return 0
    fi

    validate_path "$node_dir" "node directory" || return 0

    record_manifest "node" "$node_dir"

    # Remove node symlinks from ~/.local/bin
    for bin in node npm npx; do
        local link="$HOME/.local/bin/$bin"
        if [ -L "$link" ]; then
            local target
            target="$(readlink "$link" 2>/dev/null || echo "")"
            if echo "$target" | grep -q "$HERMES_HOME/node"; then
                rm -f "$link"
                audit_log "REMOVED symlink=$link"
            fi
        fi
    done

    rm -rf "$node_dir"
    log_success "Removed Node.js installation: $node_dir"
    audit_log "REMOVED node_dir=$node_dir"
}

remove_config() {
    if [ ! -d "$HERMES_HOME" ]; then
        log_info "Hermes data directory not found: $HERMES_HOME"
        return 0
    fi

    validate_path "$HERMES_HOME" "hermes home" || return 0

    # Safety check: refuse to remove if HERMES_HOME is home root or very short path
    case "$HERMES_HOME" in
        "$HOME"|"/"|"/home"|"/Users"|"/root"|"$HOME/")
            log_error "Refusing to remove $HERMES_HOME — appears to be a system directory"
            audit_log "REMOVE_REFUSED path=$HERMES_HOME reason=system_directory"
            return 1
            ;;
    esac

    # Make sure the path contains .hermes to prevent accidental deletion
    case "$HERMES_HOME" in
        *.hermes*|.hermes)
            ;;
        *)
            log_warn "HERMES_HOME ($HERMES_HOME) does not contain '.hermes' — extra confirmation required"
            if ! prompt_yes_no "Remove $HERMES_HOME? This does not look like a standard Hermes directory." "no"; then
                log_info "Skipped removing $HERMES_HOME"
                return 0
            fi
            ;;
    esac

    record_manifest "config" "$HERMES_HOME"

    rm -rf "$HERMES_HOME"
    log_success "Removed data directory: $HERMES_HOME"
    audit_log "REMOVED config_dir=$HERMES_HOME"
}

remove_path_entries() {
    log_info "Cleaning PATH entries from shell configs..."

    local shell_configs=()
    local login_shell
    login_shell="$(basename "${SHELL:-/bin/bash}")"

    case "$login_shell" in
        zsh)
            [ -f "$HOME/.zshrc" ] && shell_configs+=("$HOME/.zshrc")
            [ -f "$HOME/.zprofile" ] && shell_configs+=("$HOME/.zprofile")
            ;;
        bash)
            [ -f "$HOME/.bashrc" ] && shell_configs+=("$HOME/.bashrc")
            [ -f "$HOME/.bash_profile" ] && shell_configs+=("$HOME/.bash_profile")
            ;;
        fish)
            # fish handled separately below
            ;;
        *)
            [ -f "$HOME/.bashrc" ] && shell_configs+=("$HOME/.bashrc")
            [ -f "$HOME/.zshrc" ] && shell_configs+=("$HOME/.zshrc")
            ;;
    esac
    [ -f "$HOME/.profile" ] && shell_configs+=("$HOME/.profile")

    local config
    for config in "${shell_configs[@]}"; do
        if grep -q 'Hermes Agent' "$config" 2>/dev/null; then
            # Remove Hermes Agent PATH comment + export lines
            local tmp_file
            tmp_file="$(mktemp 2>/dev/null || echo "/tmp/hermes-uninstall-$$.tmp")"
            grep -v 'Hermes Agent' "$config" | grep -v 'export PATH=.*\.local/bin' | grep -v 'export PATH=.*usr/local/bin' > "$tmp_file" 2>/dev/null || true
            # Only overwrite if we got valid content
            if [ -s "$tmp_file" ]; then
                mv "$tmp_file" "$config"
                log_success "Cleaned PATH entries in $config"
                audit_log "PATH_CLEANED file=$config"
            else
                rm -f "$tmp_file"
            fi
        fi
    done

    # fish config
    local fish_config="$HOME/.config/fish/config.fish"
    if [ -f "$fish_config" ] && grep -q 'Hermes Agent' "$fish_config" 2>/dev/null; then
        local tmp_file
        tmp_file="$(mktemp 2>/dev/null || echo "/tmp/hermes-uninstall-fish-$$.tmp")"
        grep -v 'Hermes Agent' "$fish_config" | grep -v 'fish_add_path.*\.local/bin' > "$tmp_file" 2>/dev/null || true
        if [ -s "$tmp_file" ]; then
            mv "$tmp_file" "$fish_config"
            log_success "Cleaned PATH entries in $fish_config"
            audit_log "PATH_CLEANED file=$fish_config shell=fish"
        else
            rm -f "$tmp_file"
        fi
    fi
}

record_manifest() {
    local category="$1"
    local path="$2"

    local manifest_file="$HERMES_HOME/.uninstall-manifest"
    local timestamp
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date +%s)"

    echo "[$timestamp] [uninstall-$UNINSTALL_ID] category=$category path=$path" >> "$manifest_file" 2>/dev/null || true
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo ""
    echo -e "${MAGENTA}${BOLD}"
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│              Hermes Agent Uninstaller                   │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo -e "${NC}"

    audit_log "UNINSTALL_START id=$UNINSTALL_ID"

    detect_install_layout

    # Show what will be removed
    echo ""
    echo -e "${YELLOW}${BOLD}The following will be removed:${NC}"
    echo ""

    if [ "$REMOVE_GATEWAY" = true ]; then
        echo -e "  ${RED}•${NC} Gateway service (stopped + uninstalled)"
    fi
    if [ "$REMOVE_COMMAND" = true ]; then
        echo -e "  ${RED}•${NC} Command symlink: ${COMMAND_LINK_DIR:-~/.local/bin}/hermes"
    fi
    if [ "$REMOVE_CODE" = true ]; then
        echo -e "  ${RED}•${NC} Code + venv: ${INSTALL_DIR:-not found}"
    fi
    if [ "$REMOVE_NODE" = true ]; then
        echo -e "  ${RED}•${NC} Node.js: $HERMES_HOME/node/"
    fi
    if [ "$REMOVE_CONFIG" = true ]; then
        echo -e "  ${RED}•${NC} ALL data: $HERMES_HOME/"
    fi
    if [ "$REMOVE_CODE" = true ]; then
        echo -e "  ${RED}•${NC} PATH entries in shell configs"
    fi

    echo ""

    if [ "$REMOVE_CONFIG" = true ]; then
        echo -e "${RED}${BOLD}WARNING: --config will remove ALL user data including sessions, memories, skills, and API keys!${NC}"
        echo ""
    fi

    # Confirm
    if ! prompt_yes_no "Proceed with removal?" "no"; then
        log_info "Uninstall cancelled"
        audit_log "UNINSTALL_CANCELLED"
        exit 0
    fi

    # Execute removal steps
    if [ "$REMOVE_GATEWAY" = true ]; then
        stop_gateway
    fi

    if [ "$REMOVE_COMMAND" = true ]; then
        remove_command_symlink
    fi

    if [ "$REMOVE_CODE" = true ]; then
        remove_code
        remove_path_entries
    fi

    if [ "$REMOVE_NODE" = true ]; then
        remove_node
    fi

    if [ "$REMOVE_CONFIG" = true ]; then
        remove_config
    fi

    # Summary
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│              ✓ Uninstall Complete!                      │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
    echo ""

    echo -e "${CYAN}Removed:${NC}"
    [ "$REMOVE_GATEWAY" = true ] && echo "  Gateway service"
    [ "$REMOVE_COMMAND" = true ] && echo "  Command symlink"
    [ "$REMOVE_CODE" = true ]    && echo "  Code + venv"
    [ "$REMOVE_NODE" = true ]    && echo "  Node.js managed install"
    [ "$REMOVE_CONFIG" = true ]  && echo "  All data ($HERMES_HOME/)"
    [ "$REMOVE_CODE" = true ]    && echo "  PATH entries in shell configs"
    echo ""

    if [ "$REMOVE_CONFIG" = false ]; then
        echo -e "${YELLOW}Note: User data preserved at $HERMES_HOME/${NC}"
        echo -e "${YELLOW}Remove manually: rm -rf $HERMES_HOME/${NC}"
        echo ""
    fi

    echo -e "${YELLOW}Reload your shell to update PATH.${NC}"
    echo ""

    # Write manifest even if HERMES_HOME was removed (best effort)
    if [ -d "$HERMES_HOME" ]; then
        echo "Uninstall ID: $UNINSTALL_ID" >> "$HERMES_HOME/.uninstall-manifest" 2>/dev/null || true
    fi

    audit_log "UNINSTALL_COMPLETE id=$UNINSTALL_ID"
}

main "$@"