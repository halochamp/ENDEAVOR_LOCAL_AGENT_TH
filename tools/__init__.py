# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""V2 tools — LangChain @tool functions for create_react_agent"""
from .web_search import web_search
from .bash import bash
from .bash_bg import bash_bg
from .python_exec import python_exec
from .plot import plot
from .read_file import read_file
from .edit import edit
from .create_plan import create_plan
from .browse_url import browse_url
from .browser_use_tool import browser_use
from .recall_web import recall_web
from .remember import remember
from .read_image import read_image
from .speak import speak
from .awake import awake
from .computer_use import computer
from .mcp_client import mcp_list_tools, mcp_call_tool, mcp_add_server, mcp_remove_server
from .skill_tools.research_orchestrator import research_orchestrator

ALL_TOOLS = [
    web_search, bash, bash_bg, python_exec, plot, read_file, edit, create_plan,
    browse_url, browser_use, recall_web, remember, read_image, computer,
    awake, speak,
    mcp_list_tools, mcp_call_tool, mcp_add_server, mcp_remove_server,
]

# Tools bound only when their matching skill mode is active.
# Add a new entry here for any future skill-only tool — no other code change needed.
SKILL_TOOLS: dict[str, list] = {
    "research": [research_orchestrator],
}
