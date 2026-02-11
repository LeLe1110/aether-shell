#!/usr/bin/env bash
set -euo pipefail

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    echo "Usage: ./start.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -r, --reinstall    重新安装依赖 (pip install -e .)"
    echo "  -c, --clean        清除虚拟环境后重建"
    echo "  -h, --help         显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  ./start.sh              # 正常启动"
    echo "  ./start.sh -r           # 重新安装依赖后启动"
    echo "  ./start.sh --clean      # 清除并重建虚拟环境后启动"
}

REINSTALL=false
CLEAN=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--reinstall)
            REINSTALL=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=== Starting aether-bot gateway ===${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

# 清除虚拟环境
if [[ "$CLEAN" == true ]] && [[ -d "$VENV_DIR" ]]; then
    echo -e "${YELLOW}Removing virtual environment...${NC}"
    rm -rf "$VENV_DIR"
fi

# 检查/创建虚拟环境
if [[ ! -x "$VENV_PY" ]]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
    REINSTALL=true  # 新环境必须安装依赖
fi

# 安装依赖
if [[ "$REINSTALL" == true ]]; then
    echo -e "${YELLOW}Installing dependencies (pip install -e .)...${NC}"
    "$VENV_DIR/bin/pip" install -e .
fi

# 检查配置文件
CONFIG_FILE="$HOME/.aether-bot/config.json"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}Config file not found: $CONFIG_FILE${NC}"
    echo -e "${YELLOW}Run 'nanobot onboard' first to initialize configuration.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Virtual environment ready${NC}"
echo -e "${GREEN}✓ Config file found${NC}"
echo -e "${BLUE}Starting gateway with live logs...${NC}"
echo ""

# 直接运行，输出到控制台
exec "$VENV_PY" -m nanobot gateway
