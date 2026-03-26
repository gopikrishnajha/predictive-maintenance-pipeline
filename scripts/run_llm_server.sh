#!/bin/bash
# Start/Stop/Status script for PACE LLM Server
# Keeps the LLM loaded in memory for persistent KV cache

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$PROJECT_ROOT/.llm_server.pid"
LOG_FILE="$PROJECT_ROOT/out/llm_server.log"

cd "$PROJECT_ROOT" || exit 1

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function start_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  LLM server is already running (PID: $PID)${NC}"
            echo "   Use: $0 stop   to stop it"
            return 1
        else
            # Stale PID file
            rm -f "$PID_FILE"
        fi
    fi
    
    echo -e "${GREEN}🚀 Starting LLM server...${NC}"
    
    # Check if conda environment is activated
    if [[ "$CONDA_DEFAULT_ENV" != "pace" ]]; then
        echo -e "${YELLOW}⚠️  Warning: 'pace' conda environment not activated${NC}"
        echo "   Activate it with: conda activate pace"
        return 1
    fi
    
    # Create output directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Check dependencies
    if ! python -c "import fastapi" 2>/dev/null; then
        echo -e "${RED}❌ FastAPI not installed${NC}"
        echo "   Install with: pip install fastapi uvicorn"
        return 1
    fi
    
    # Start server in background
    nohup python scripts/llm_server.py > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    
    # Save PID
    echo $SERVER_PID > "$PID_FILE"
    
    # Wait a bit and check if it started successfully
    sleep 3
    
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        echo -e "${GREEN}✅ LLM server started successfully (PID: $SERVER_PID)${NC}"
        echo "   Log file: $LOG_FILE"
        echo "   API docs: http://localhost:8000/docs"
        echo ""
        echo "   Check status: $0 status"
        echo "   View logs:    $0 logs"
        echo "   Stop server:  $0 stop"
        return 0
    else
        echo -e "${RED}❌ Server failed to start. Check logs:${NC}"
        echo "   tail -n 20 $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

function stop_server() {
    if [ ! -f "$PID_FILE" ]; then
        echo -e "${YELLOW}⚠️  LLM server is not running (no PID file)${NC}"
        return 1
    fi
    
    PID=$(cat "$PID_FILE")
    
    if ! ps -p $PID > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  LLM server is not running (stale PID)${NC}"
        rm -f "$PID_FILE"
        return 1
    fi
    
    echo -e "${YELLOW}🛑 Stopping LLM server (PID: $PID)...${NC}"
    kill $PID
    
    # Wait for process to stop
    for _ in {1..10}; do
        if ! ps -p $PID > /dev/null 2>&1; then
            echo -e "${GREEN}✅ LLM server stopped${NC}"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
    done
    
    # Force kill if still running
    echo -e "${YELLOW}⚠️  Force killing server...${NC}"
    kill -9 $PID 2>/dev/null
    rm -f "$PID_FILE"
    echo -e "${GREEN}✅ LLM server killed${NC}"
}

function status_server() {
    if [ ! -f "$PID_FILE" ]; then
        echo -e "${RED}❌ LLM server is not running${NC}"
        echo "   Start with: $0 start"
        return 1
    fi
    
    PID=$(cat "$PID_FILE")
    
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${GREEN}✅ LLM server is running (PID: $PID)${NC}"
        
        # Try to get health status
        if command -v curl &> /dev/null; then
            echo ""
            echo "Server health:"
            curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "  (Health check unavailable)"
        fi
        
        echo ""
        echo "API endpoints:"
        echo "  • Health:  http://localhost:8000/health"
        echo "  • Docs:    http://localhost:8000/docs"
        echo "  • OpenAPI: http://localhost:8000/openapi.json"
        return 0
    else
        echo -e "${RED}❌ LLM server is not running (stale PID)${NC}"
        rm -f "$PID_FILE"
        return 1
    fi
}

function show_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${YELLOW}⚠️  No log file found: $LOG_FILE${NC}"
        return 1
    fi
    
    echo -e "${GREEN}📄 Last 50 lines of LLM server logs:${NC}"
    echo "---"
    tail -n 50 "$LOG_FILE"
}

function restart_server() {
    echo -e "${YELLOW}🔄 Restarting LLM server...${NC}"
    stop_server
    sleep 2
    start_server
}

# Main script
case "${1:-status}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        status_server
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "PACE LLM Server Control Script"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the LLM server (keeps model loaded)"
        echo "  stop     - Stop the LLM server"
        echo "  restart  - Restart the LLM server"
        echo "  status   - Check if server is running"
        echo "  logs     - Show recent server logs"
        echo ""
        echo "The server keeps the LLM loaded in memory for fast inference"
        echo "and preserves KV cache across requests for better performance."
        exit 1
        ;;
esac
